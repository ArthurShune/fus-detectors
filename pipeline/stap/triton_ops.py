"""
Optional Triton kernels for STAP latency hot paths.

These are only used when:
  - running on CUDA, and
  - Triton is available, and
  - the corresponding env var toggle is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton missing / CPU-only env
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return bool(_TRITON_AVAILABLE)


@dataclass(frozen=True)
class TylerWeightsConfig:
    block_p: int = 256
    num_warps: int = 4
    w_max: float = 1e6


if _TRITON_AVAILABLE:  # pragma: no cover - exercised only in CUDA envs

    @triton.jit
    def _tyler_weights_scale_kernel(
        Y_ptr,  # fp32 view: (B, Lt, P, 2)
        S_ptr,  # fp32 view: (B, Lt, P, 2)
        O_ptr,  # fp32 view: (B, Lt, P, 2)
        P,
        Lt: tl.constexpr,
        eps,
        w_max,
        stride_y_b,
        stride_y_lt,
        stride_y_p,
        stride_y_c,
        stride_s_b,
        stride_s_lt,
        stride_s_p,
        stride_s_c,
        stride_o_b,
        stride_o_lt,
        stride_o_p,
        stride_o_c,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mask_p = p < P

        # q[p] = sum_{lt,c} Y[b,lt,p,c]^2
        q = tl.zeros((BLOCK_P,), dtype=tl.float32)
        for lt_idx in range(0, Lt):
            base_y = pid_b * stride_y_b + lt_idx * stride_y_lt + p * stride_y_p
            y0 = tl.load(Y_ptr + base_y + 0 * stride_y_c, mask=mask_p, other=0.0)
            y1 = tl.load(Y_ptr + base_y + 1 * stride_y_c, mask=mask_p, other=0.0)
            q += y0 * y0 + y1 * y1
        q = tl.maximum(q, eps)
        w = (tl.full((BLOCK_P,), Lt, tl.float32) / q).to(tl.float32)
        w = tl.minimum(w, w_max)

        # O[b,lt,p,c] = S[b,lt,p,c] * w[p]
        for lt_idx in range(0, Lt):
            base_s = pid_b * stride_s_b + lt_idx * stride_s_lt + p * stride_s_p
            s0 = tl.load(S_ptr + base_s + 0 * stride_s_c, mask=mask_p, other=0.0)
            s1 = tl.load(S_ptr + base_s + 1 * stride_s_c, mask=mask_p, other=0.0)
            out0 = s0 * w
            out1 = s1 * w
            base_o = pid_b * stride_o_b + lt_idx * stride_o_lt + p * stride_o_p
            tl.store(O_ptr + base_o + 0 * stride_o_c, out0, mask=mask_p)
            tl.store(O_ptr + base_o + 1 * stride_o_c, out1, mask=mask_p)


def tyler_weights_scale_triton(
    Y_B_Lt_P: "torch.Tensor",
    S_B_Lt_P: "torch.Tensor",
    out_B_Lt_P: "torch.Tensor",
    *,
    Lt: int,
    eps: float,
    cfg: TylerWeightsConfig = TylerWeightsConfig(),
):
    """
    Fused Tyler weights + snapshot scaling:

      q[b,p] = sum_lt |Y[b,lt,p]|^2
      w[b,p] = min(Lt / max(q, eps), w_max)
      out[b,lt,p] = S[b,lt,p] * w[b,p]

    Notes
    -----
    - Input/output are complex64 tensors with shape (B, Lt, P).
    - Implementation operates on the fp32 `view_as_real` storage.
    - `Y_B_Lt_P` and `out_B_Lt_P` are allowed to alias (in-place overwrite),
      because the kernel fully reduces `Y` before writing `out`.
    """
    import torch  # local import; torch is an unconditional repo dependency

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if not (Y_B_Lt_P.is_cuda and S_B_Lt_P.is_cuda and out_B_Lt_P.is_cuda):
        raise ValueError("Triton Tyler weights requires CUDA tensors.")
    if Y_B_Lt_P.dtype != torch.complex64 or S_B_Lt_P.dtype != torch.complex64 or out_B_Lt_P.dtype != torch.complex64:
        raise ValueError("Triton Tyler weights currently supports complex64 only.")
    if Y_B_Lt_P.shape != S_B_Lt_P.shape or Y_B_Lt_P.shape != out_B_Lt_P.shape:
        raise ValueError("Y/S/out must have identical shapes (B,Lt,P).")
    if Y_B_Lt_P.ndim != 3:
        raise ValueError(f"Expected (B,Lt,P), got {tuple(Y_B_Lt_P.shape)}")
    B, Lt_in, P = (int(Y_B_Lt_P.shape[0]), int(Y_B_Lt_P.shape[1]), int(Y_B_Lt_P.shape[2]))
    if int(Lt) != int(Lt_in):
        raise ValueError(f"Lt mismatch: arg Lt={int(Lt)} tensor Lt={int(Lt_in)}")

    Y = torch.view_as_real(Y_B_Lt_P)  # (B,Lt,P,2) float32 view
    S = torch.view_as_real(S_B_Lt_P)
    O = torch.view_as_real(out_B_Lt_P)

    # Strides are in fp32 elements for the view tensors.
    syb, sylt, syp, syc = (int(s) for s in Y.stride())
    ssb, sslt, ssp, ssc = (int(s) for s in S.stride())
    sob, solt, sop, soc = (int(s) for s in O.stride())

    grid = (B, triton.cdiv(P, int(cfg.block_p)))
    _tyler_weights_scale_kernel[grid](
        Y,
        S,
        O,
        P=P,
        Lt=int(Lt),
        eps=float(eps),
        w_max=float(cfg.w_max),
        stride_y_b=syb,
        stride_y_lt=sylt,
        stride_y_p=syp,
        stride_y_c=syc,
        stride_s_b=ssb,
        stride_s_lt=sslt,
        stride_s_p=ssp,
        stride_s_c=ssc,
        stride_o_b=sob,
        stride_o_lt=solt,
        stride_o_p=sop,
        stride_o_c=soc,
        BLOCK_P=int(cfg.block_p),
        num_warps=int(cfg.num_warps),
    )
