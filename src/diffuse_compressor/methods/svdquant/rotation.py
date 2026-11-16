"""Input-basis rotations for SVDQuant W4A4 residuals.

Basis convention
----------------
For one projector with weight ``W [N, K]``, smoothing ``s [K]`` and low-rank
branch ``L1 [r, K]`` (down) / ``L2 [N, r]`` (up), the W4A4 runtime computes::

    x_s = x / s
    y   = x_s @ L1^T @ L2^T  +  Q(x_s @ H) @ Q(R @ H)^T  +  b
    R   = W @ diag(s) - L2 @ L1

``H = blockdiag(H64, ..., H64)`` along ``K`` and ``H64`` is the Sylvester
Walsh-Hadamard matrix divided by 8 (``scipy.linalg.hadamard(64) / 8``). It is
symmetric and orthonormal, so ``H^-1 = H^T = H`` and rotating twice is the
identity. Without quantization the rotated term equals ``x_s @ R^T`` exactly.

Rotated (the quantized basis):
    * the residual weight ``R`` along its input dimension, before 4-bit rounding;
    * the smoothed activation ``x_s`` entering the 4-bit activation quantizer.

Not rotated:
    * the raw input ``x``, the smoothing factors ``s`` and the bias;
    * the low-rank branch ``L1``/``L2`` and its input (the kernel runs LoRA-down
      on the unrotated ``x_s``);
    * the output ``y``.

Quantization search code works in the original basis and only enters the
rotated basis around rounding, via :meth:`InputRotation.quantize_in_basis`:
``Q_rot(t) = rotate(Q(rotate(t)))``. Because ``H`` is orthonormal,
``Q_rot(x_s) @ Q_rot(R)^T == Q(x_s H) @ Q(R H)^T``, which is what the kernel
computes. The default :data:`IDENTITY` rotation makes every hook a no-op that
returns its input unchanged, so the unrotated flow is the same code path.

Not supported: random sign flips (``D @ H``), rotating the LoRA-down input,
fused-GLU activation quantization, and Conv2d targets. Inference needs the
nunchaku-lite W4A4 kernel built with ``hadamard=True`` (CUDA only).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping

import torch

from ...config import RotationSpec

TensorFn = Callable[[torch.Tensor], torch.Tensor]

HADAMARD_BLOCK_SIZE = 64


@lru_cache(maxsize=None)
def _hadamard_matrix_cached(
    block_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(
            f"Hadamard block_size must be a positive power of two, got {block_size}"
        )
    # Sylvester construction, H_2n = [[H_n, H_n], [H_n, -H_n]] = kron(H_2, H_n),
    # which is the ordering scipy.linalg.hadamard uses.
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < block_size:
        matrix = torch.kron(base, matrix)
    # 1/sqrt(n) is a power of two for n = 4**k (1/8 for n = 64), so this is exact.
    matrix = matrix / matrix.shape[0] ** 0.5
    return matrix.to(device=device, dtype=dtype)


def hadamard_matrix(
    block_size: int = HADAMARD_BLOCK_SIZE,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the orthonormal Sylvester Hadamard matrix ``hadamard(n) / sqrt(n)``.

    The result is cached per ``(block_size, device, dtype)`` and shared between
    callers, so it must not be modified in place.
    """

    return _hadamard_matrix_cached(block_size, torch.device(device or "cpu"), dtype)


def check_rotation_compatible(
    rotation: RotationSpec,
    *,
    in_features: int,
    kind: str = "linear",
    name: str = "target",
    precision: str = "int4",
) -> None:
    """Raise ``ValueError`` when a target cannot use ``rotation``.

    Incompatible targets fail instead of being skipped silently: a skipped
    layer would need its own runtime flag. Exclude such layers from rotation
    explicitly, for example by quantizing them in a separate run or with AWQ.
    """

    if kind != "linear":
        raise ValueError(
            f"Hadamard rotation supports linear targets only; {name!r} is a {kind!r} target"
        )
    if in_features % rotation.block_size != 0:
        raise ValueError(
            f"Hadamard rotation requires in_features divisible by {rotation.block_size}; "
            f"{name!r} has in_features={in_features}"
        )
    if precision == "fp4" and not rotation.allow_fp4:
        raise ValueError(
            f"Hadamard rotation is validated for INT4 only; {name!r} is an NVFP4 (fp4) target. "
            "Pass RotationSpec(allow_fp4=True) to experiment."
        )


class InputRotation:
    """Identity transform of a projector's quantized input basis (the default).

    Every method returns its input unchanged, so a flow that threads this
    object through behaves exactly like one that never rotates.
    """

    @property
    def enabled(self) -> bool:
        """Whether this transform changes the quantized basis."""

        return False

    def rotate_input_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rotate a weight-like tensor ``[..., K]`` along its input (last) dimension."""

        return tensor

    def rotate_activations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rotate activation rows ``[..., K]`` along their channel (last) dimension."""

        return self.rotate_input_dim(tensor)

    def quantize_in_basis(
        self, tensor: torch.Tensor, quantize: TensorFn
    ) -> torch.Tensor:
        """Fake-quantize ``tensor`` in the rotated basis and return it in the original basis."""

        return quantize(tensor)

    def wrap(self, quantize: TensorFn | None) -> TensorFn | None:
        """Return ``quantize`` lifted to run in the rotated basis."""

        return quantize

    def metadata(self) -> dict[str, object] | None:
        """Return the checkpoint schema for this rotation, or ``None`` for identity."""

        return None


IDENTITY = InputRotation()


class BlockHadamardRotation(InputRotation):
    """Block-diagonal Walsh-Hadamard rotation ``blockdiag(H_b, ..., H_b)`` along ``K``.

    Args:
        block_size: Hadamard block width ``b``; ``K`` must be divisible by it.
    """

    def __init__(self, block_size: int = HADAMARD_BLOCK_SIZE) -> None:
        hadamard_matrix(block_size)  # validate eagerly
        self.block_size = block_size

    @property
    def enabled(self) -> bool:
        return True

    def rotate_input_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        in_features = tensor.shape[-1]
        if in_features % self.block_size != 0:
            raise ValueError(
                f"Hadamard rotation requires a last dimension divisible by {self.block_size}, got {in_features}"
            )
        # Rotate in float32 and round once to the input dtype. The kernel's
        # fp16/bf16 FWHT rounds per butterfly stage, so it matches this to
        # within half-precision tolerance, not bit-exactly.
        matrix = hadamard_matrix(
            self.block_size, device=tensor.device, dtype=torch.float32
        )
        blocks = tensor.float().reshape(
            *tensor.shape[:-1], in_features // self.block_size, self.block_size
        )
        return (blocks @ matrix).reshape(tensor.shape).to(dtype=tensor.dtype)

    def quantize_in_basis(
        self, tensor: torch.Tensor, quantize: TensorFn
    ) -> torch.Tensor:
        # H is symmetric and orthonormal, so the inverse rotation is the rotation itself.
        return self.rotate_input_dim(quantize(self.rotate_input_dim(tensor)))

    def wrap(self, quantize: TensorFn | None) -> TensorFn | None:
        if quantize is None:
            return None
        return lambda tensor: self.quantize_in_basis(tensor, quantize)

    def metadata(self) -> dict[str, object]:
        return {"type": "hadamard", "block_size": self.block_size}

    def __repr__(self) -> str:
        return f"BlockHadamardRotation(block_size={self.block_size})"


def resolve_input_rotation(
    rotation: RotationSpec | None,
    *,
    in_features: int,
    kind: str = "linear",
    name: str = "target",
    precision: str = "int4",
) -> InputRotation:
    """Build the input-basis transform for one target, validating compatibility.

    Args:
        rotation: Requested rotation, or ``None`` for the identity basis.
        in_features: Target input dimension ``K``.
        kind: Target kind; only ``"linear"`` supports rotation.
        name: Target name used in error messages.
        precision: Target residual precision; ``"fp4"`` requires
            ``rotation.allow_fp4``.

    Returns:
        :data:`IDENTITY` or a :class:`BlockHadamardRotation`.
    """

    if rotation is None:
        return IDENTITY
    check_rotation_compatible(rotation, in_features=in_features, kind=kind, name=name, precision=precision)
    return BlockHadamardRotation(rotation.block_size)


def input_rotation_from_metadata(
    metadata: Mapping[str, object] | None,
) -> InputRotation:
    """Rebuild a rotation from its :meth:`InputRotation.metadata` checkpoint form."""

    if metadata is None:
        return IDENTITY
    if metadata.get("type") != "hadamard":
        raise ValueError(f"Unsupported input rotation metadata: {dict(metadata)!r}")
    return BlockHadamardRotation(int(metadata.get("block_size", HADAMARD_BLOCK_SIZE)))  # type: ignore[arg-type]
