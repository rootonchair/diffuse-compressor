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
equals ``x_s @ R^T`` exactly. Checkpoints declare the transform by name and
store no matrix: ``{"type": "hadamard"}`` means exactly ``T_i = H64 / 8``, the
Sylvester Walsh-Hadamard matrix divided by ``sqrt(64)``
(``scipy.linalg.hadamard(64) / 8``, also symmetric), on every contiguous
64-channel block in logical channel order. ``nunchaku_lite.ops.hadamard_rotate``
is the reference transform. A different transform needs a new name.

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
fused-GLU activation quantization, and Conv2d targets. The only rotation
exposed through :class:`RotationSpec` is Hadamard; fast inference needs
nunchaku-lite's separate Hadamard quantize op
(``quantize_w4a4_act_hadamard_fuse_lora``, CUDA only).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch

from ...config import RotationSpec

TensorFn = Callable[[torch.Tensor], torch.Tensor]

HADAMARD_BLOCK_SIZE = 64
HADAMARD_METADATA: Mapping[str, object] = {"type": "hadamard"}
"""Checkpoint form of the only exportable rotation: blockdiag(H64 / 8) on ``x / smooth``."""


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


IDENTITY = InputRotation()


class BlockMatrixRotation(InputRotation):
    """Orthogonal block-diagonal input transform ``blockdiag(T_1, ..., T_{K/b})`` along ``K``.

    This is the basis math only. It has no checkpoint form: an exported rotation
    must have a name loaders know (see :class:`BlockHadamardRotation`).

    Args:
        blocks: ``[b, b]`` matrix shared by every block, or ``[K/b, b, b]`` with one
            matrix per block. Must be orthogonal (``T @ T^T == I``).
    """

    def __init__(self, blocks: torch.Tensor) -> None:
        blocks = blocks.detach().to(dtype=torch.float32)
        if blocks.dim() not in (2, 3) or blocks.shape[-1] != blocks.shape[-2]:
            raise ValueError(f"input transform must be [b, b] or [K/b, b, b], got {tuple(blocks.shape)}")
        eye = torch.eye(blocks.shape[-1], dtype=torch.float32, device=blocks.device)
        if (blocks @ blocks.transpose(-1, -2) - eye).abs().max() > 1e-3:
            raise ValueError("input transform blocks must be orthogonal (T @ T^T == I)")
        self.blocks = blocks.cpu()
        self.block_size = int(blocks.shape[-1])
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
        raise NotImplementedError(f"{type(self).__name__} has no checkpoint form; export a named rotation")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(block_size={self.block_size}, shared={self.shared})"


class BlockHadamardRotation(BlockMatrixRotation):
    """Shared Sylvester Hadamard blocks ``hadamard(b) / sqrt(b)``.

    Only ``block_size=64`` has a checkpoint form (``{"type": "hadamard"}``).

    Args:
        block_size: Hadamard block width ``b``; ``K`` must be divisible by it.
    """

    def __init__(self, block_size: int = HADAMARD_BLOCK_SIZE) -> None:
        super().__init__(hadamard_matrix(block_size))

    def metadata(self) -> dict[str, object]:
        if self.block_size != HADAMARD_BLOCK_SIZE:
            raise ValueError(f'{{"type": "hadamard"}} means 64-wide blocks; this rotation uses {self.block_size}')
        return dict(HADAMARD_METADATA)

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


def input_rotation_from_metadata(metadata: Mapping[str, object] | None) -> InputRotation:
    """Rebuild a rotation from its checkpoint form.

    Args:
        metadata: Target ``rotation`` metadata, or ``None`` for an unrotated target.
            Only ``{"type": "hadamard"}``, with no other fields, is accepted.

    Returns:
        :data:`IDENTITY` or a 64-wide :class:`BlockHadamardRotation`.
    """

    if metadata is None:
        return IDENTITY
    if dict(metadata) == HADAMARD_METADATA:
        return BlockHadamardRotation(HADAMARD_BLOCK_SIZE)
    if metadata.get("type") == "block_matrix":
        raise ValueError(
            "Rotation metadata {'type': 'block_matrix'} is the pre-release format with stored input_transform "
            "tensors; re-export the checkpoint with {'type': 'hadamard'}"
        )
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
