"""Input-basis rotations for SVDQuant W4A4 residuals.

Basis convention
----------------
For one projector with weight ``W [N, K]``, smoothing ``s [K]`` and low-rank
branch ``L1 [r, K]`` (down) / ``L2 [N, r]`` (up), the W4A4 runtime computes::

    x_s = x / s
    y   = x_s @ L1^T @ L2^T  +  Q(x_s @ T) @ Q(R @ T)^T  +  b
    R   = W @ diag(s) - L2 @ L1

``T = blockdiag(T_1, ..., T_{K/b})`` is an orthogonal block-diagonal input
transform along ``K`` (``T^-1 = T^T``), so without quantization the rotated term
equals ``x_s @ R^T`` exactly. The transform is data: checkpoints store its
blocks as the ``input_transform`` tensor (``[b, b]`` shared by every block, or
``[K/b, b, b]``) next to ``{"type": "block_matrix", ...}`` metadata, so loaders
and kernels apply whatever matrix they are given instead of a fixed catalogue.
The default generator is the Sylvester Walsh-Hadamard matrix divided by
``sqrt(b)`` (``scipy.linalg.hadamard(64) / 8``), which is also symmetric.

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
``Q_rot(t) = Q(t @ T) @ T^T``. Because ``T`` is orthogonal,
``Q_rot(x_s) @ Q_rot(R)^T == Q(x_s T) @ Q(R T)^T``, which is what the kernel
computes. The default :data:`IDENTITY` rotation makes every hook a no-op that
returns its input unchanged, so the unrotated flow is the same code path.

Not supported: non-orthogonal transforms, rotating the LoRA-down input,
fused-GLU activation quantization, and Conv2d targets. The only generator
exposed through :class:`RotationSpec` today is Hadamard; inference needs the
nunchaku-lite W4A4 kernel's activation transform (``hadamard=True`` in the
current build, CUDA only).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping, Sequence

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
    signed_activations: bool = True,
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
    if not signed_activations:
        raise ValueError(
            f"Input rotation requires a signed activation path; {name!r} quantizes activations unsigned "
            "(shift_activations=True). Set SvdqTargetQuant(rotation=False) on it to export it unrotated."
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

    def matrix(self) -> torch.Tensor | None:
        """Return the transform blocks to store as ``input_transform``, or ``None``."""

        return None


IDENTITY = InputRotation()


class BlockMatrixRotation(InputRotation):
    """Orthogonal block-diagonal input transform ``blockdiag(T_1, ..., T_{K/b})`` along ``K``.

    Args:
        blocks: ``[b, b]`` matrix shared by every block, or ``[K/b, b, b]`` with one
            matrix per block. Must be orthogonal (``T @ T^T == I``).
        generator: Optional name of the construction (e.g. ``"hadamard"``),
            recorded as a hint for loaders with a specialized fast path.
    """

    def __init__(self, blocks: torch.Tensor, *, generator: str | None = None) -> None:
        blocks = blocks.detach().to(dtype=torch.float32)
        if blocks.dim() not in (2, 3) or blocks.shape[-1] != blocks.shape[-2]:
            raise ValueError(f"input transform must be [b, b] or [K/b, b, b], got {tuple(blocks.shape)}")
        eye = torch.eye(blocks.shape[-1], dtype=torch.float32, device=blocks.device)
        if (blocks @ blocks.transpose(-1, -2) - eye).abs().max() > 1e-3:
            raise ValueError("input transform blocks must be orthogonal (T @ T^T == I)")
        self.blocks = blocks.cpu()
        self.block_size = int(blocks.shape[-1])
        self.generator = generator
        self._device_cache: dict[torch.device, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return True

    @property
    def shared(self) -> bool:
        return self.blocks.dim() == 2

    def _blocks_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._device_cache:
            self._device_cache[device] = self.blocks.to(device=device)
        return self._device_cache[device]

    def _apply(self, tensor: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        in_features = tensor.shape[-1]
        if in_features % self.block_size != 0:
            raise ValueError(
                f"Input transform requires a last dimension divisible by {self.block_size}, got {in_features}"
            )
        blocks = self._blocks_on(tensor.device)
        if transpose:
            blocks = blocks.transpose(-1, -2)
        if not self.shared and blocks.shape[0] != in_features // self.block_size:
            raise ValueError(
                f"Per-block input transform has {blocks.shape[0]} blocks, expected {in_features // self.block_size}"
            )
        # Transform in float32 and round once to the input dtype. The kernel's
        # half-precision transform rounds internally, so it matches this to
        # within half-precision tolerance, not bit-exactly.
        grouped = tensor.float().reshape(*tensor.shape[:-1], in_features // self.block_size, self.block_size)
        if self.shared:
            out = grouped @ blocks
        else:
            out = torch.einsum("...gi,gij->...gj", grouped, blocks)
        return out.reshape(tensor.shape).to(dtype=tensor.dtype)

    def rotate_input_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, transpose=False)

    def unrotate_input_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ``T^T = T^-1`` along the last dimension."""

        return self._apply(tensor, transpose=True)

    def quantize_in_basis(
        self, tensor: torch.Tensor, quantize: TensorFn
    ) -> torch.Tensor:
        return self.unrotate_input_dim(quantize(self.rotate_input_dim(tensor)))

    def wrap(self, quantize: TensorFn | None) -> TensorFn | None:
        if quantize is None:
            return None
        return lambda tensor: self.quantize_in_basis(tensor, quantize)

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {"type": "block_matrix", "block_size": self.block_size, "shared": self.shared}
        if self.generator is not None:
            metadata["generator"] = self.generator
        return metadata

    def matrix(self) -> torch.Tensor:
        return self.blocks

    def __repr__(self) -> str:
        kind = self.generator or "matrix"
        return f"{type(self).__name__}({kind}, block_size={self.block_size}, shared={self.shared})"


class BlockHadamardRotation(BlockMatrixRotation):
    """Shared Sylvester Hadamard blocks ``hadamard(b) / sqrt(b)``: the default generator.

    Args:
        block_size: Hadamard block width ``b``; ``K`` must be divisible by it.
    """

    def __init__(self, block_size: int = HADAMARD_BLOCK_SIZE) -> None:
        super().__init__(hadamard_matrix(block_size), generator="hadamard")

    def __repr__(self) -> str:
        return f"BlockHadamardRotation(block_size={self.block_size})"


def resolve_input_rotation(
    rotation: RotationSpec | None,
    *,
    in_features: int,
    kind: str = "linear",
    name: str = "target",
    precision: str = "int4",
    signed_activations: bool = True,
) -> InputRotation:
    """Build the input-basis transform for one target, validating compatibility.

    Args:
        rotation: Requested rotation, or ``None`` for the identity basis.
        in_features: Target input dimension ``K``.
        kind: Target kind; only ``"linear"`` supports rotation.
        name: Target name used in error messages.
        precision: Target residual precision; ``"fp4"`` requires
            ``rotation.allow_fp4``.
        signed_activations: Whether the target quantizes activations signed;
            unsigned (shifted) targets cannot be rotated.

    Returns:
        :data:`IDENTITY` or a :class:`BlockHadamardRotation`.
    """

    if rotation is None:
        return IDENTITY
    check_rotation_compatible(
        rotation, in_features=in_features, kind=kind, name=name, precision=precision,
        signed_activations=signed_activations,
    )
    return BlockHadamardRotation(rotation.block_size)


def input_rotation_from_metadata(
    metadata: Mapping[str, object] | None,
    matrix: torch.Tensor | None = None,
) -> InputRotation:
    """Rebuild a rotation from its checkpoint form.

    Args:
        metadata: Target ``rotation`` metadata, or ``None`` for an unrotated target.
        matrix: The target's stored ``input_transform`` tensor. Required for
            ``{"type": "block_matrix"}``.

    Returns:
        :data:`IDENTITY` or a :class:`BlockMatrixRotation` built from the stored blocks.
    """

    if metadata is None:
        return IDENTITY
    kind = metadata.get("type")
    if kind == "block_matrix":
        if matrix is None:
            raise ValueError("block_matrix rotation metadata requires the target's input_transform tensor")
        generator = metadata.get("generator")
        return BlockMatrixRotation(matrix, generator=None if generator is None else str(generator))
    raise ValueError(f"Unsupported input rotation metadata: {dict(metadata)!r}")


def target_rotation_spec(rotation: RotationSpec | None, quant: object) -> RotationSpec | None:
    """Return ``rotation`` unless the target opts out with ``rotation=False``."""

    if rotation is None or getattr(quant, "rotation", None) is False:
        return None
    return rotation


def validate_rotation_targets(rotation: RotationSpec | None, targets: Sequence[object]) -> None:
    """Fail before calibration when a rotated target cannot run on a signed activation path.

    Raises:
        ValueError: Naming every shifted (unsigned) target that would be rotated.
    """

    if rotation is None:
        return
    unsigned = [
        target.export_name
        for target in targets
        if target_rotation_spec(rotation, target.quant) is not None
        and getattr(target.quant, "shift_activations", None) is True
    ]
    if unsigned:
        raise ValueError(
            "Input rotation requires a signed activation path, but these targets quantize activations unsigned "
            f"(shift_activations=True): {unsigned}. Set SvdqTargetQuant(rotation=False) on them to export them "
            "unrotated."
        )
