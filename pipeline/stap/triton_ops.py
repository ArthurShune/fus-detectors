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


@dataclass(frozen=True)
class TylerCovUpdateConfig:
    block_p: int = 256
    num_warps: int = 4


@dataclass(frozen=True)
class TylerWeightsUpdateConfig:
    block_p: int = 256
    num_warps: int = 4
    w_max: float = 1e6


@dataclass(frozen=True)
class TylerInvGemmUpdateConfig:
    block_p: int = 256
    num_warps: int = 2
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

    @triton.jit
    def _tyler_cov_update_kernel(
        X_ptr,  # fp32 view: (B, Lt, P, 2)
        O_ptr,  # fp32 view: (B, Lt, Lt, 2)
        P,
        Lt: tl.constexpr,
        inv_P,
        stride_x_b,
        stride_x_lt,
        stride_x_p,
        stride_x_c,
        stride_o_b,
        stride_o_lt0,
        stride_o_lt1,
        stride_o_c,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        lt = tl.arange(0, Lt)

        real_acc = tl.zeros((Lt, Lt), dtype=tl.float32)
        imag_acc = tl.zeros((Lt, Lt), dtype=tl.float32)

        for p0 in range(0, P, BLOCK_P):
            p = p0 + tl.arange(0, BLOCK_P)
            mask_p = p < P

            base_x = pid_b * stride_x_b + lt[:, None] * stride_x_lt + p[None, :] * stride_x_p
            xr = tl.load(
                X_ptr + base_x + 0 * stride_x_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            xi = tl.load(
                X_ptr + base_x + 1 * stride_x_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)

            # Accumulate X @ X^H (complex):
            #   real = sum_p (xr_i*xr_j + xi_i*xi_j)
            #   imag = sum_p (xi_i*xr_j - xr_i*xi_j)
            xr_i = xr[:, None, :]
            xr_j = xr[None, :, :]
            xi_i = xi[:, None, :]
            xi_j = xi[None, :, :]
            real_acc += tl.sum(xr_i * xr_j + xi_i * xi_j, axis=2)
            imag_acc += tl.sum(xi_i * xr_j - xr_i * xi_j, axis=2)

        real_acc *= inv_P
        imag_acc *= inv_P

        lt0 = lt[:, None]
        lt1 = lt[None, :]
        base_o = pid_b * stride_o_b + lt0 * stride_o_lt0 + lt1 * stride_o_lt1
        tl.store(O_ptr + base_o + 0 * stride_o_c, real_acc)
        tl.store(O_ptr + base_o + 1 * stride_o_c, imag_acc)

    @triton.jit
    def _tyler_weights_update_cov_kernel(
        Y_ptr,  # fp32 view: (B, Lt, P, 2)
        S_ptr,  # fp32 view: (B, Lt, P, 2)
        O_ptr,  # fp32 view: (B, Lt, Lt, 2)
        P,
        Lt: tl.constexpr,
        eps,
        w_max,
        inv_P,
        stride_y_b,
        stride_y_lt,
        stride_y_p,
        stride_y_c,
        stride_s_b,
        stride_s_lt,
        stride_s_p,
        stride_s_c,
        stride_o_b,
        stride_o_lt0,
        stride_o_lt1,
        stride_o_c,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        lt = tl.arange(0, Lt)[:, None]

        real_acc = tl.zeros((Lt, Lt), dtype=tl.float32)
        imag_acc = tl.zeros((Lt, Lt), dtype=tl.float32)

        for p0 in range(0, P, BLOCK_P):
            p = p0 + tl.arange(0, BLOCK_P)
            mask_p = p < P

            # q[p] = sum_lt |Y[b,lt,p]|^2
            base_y = pid_b * stride_y_b + lt * stride_y_lt + p[None, :] * stride_y_p
            y0 = tl.load(
                Y_ptr + base_y + 0 * stride_y_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            y1 = tl.load(
                Y_ptr + base_y + 1 * stride_y_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            q = tl.sum(y0 * y0 + y1 * y1, axis=0)
            q = tl.maximum(q, eps)
            w = (tl.full((BLOCK_P,), Lt, tl.float32) / q).to(tl.float32)
            w = tl.minimum(w, w_max)

            # Xw = S * w (per snapshot)
            base_s = pid_b * stride_s_b + lt * stride_s_lt + p[None, :] * stride_s_p
            s0 = tl.load(
                S_ptr + base_s + 0 * stride_s_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            s1 = tl.load(
                S_ptr + base_s + 1 * stride_s_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            xr = s0 * w[None, :]
            xi = s1 * w[None, :]

            # Accumulate Xw @ Xw^H (complex):
            #   real = sum_p (xr_i*xr_j + xi_i*xi_j)
            #   imag = sum_p (xi_i*xr_j - xr_i*xi_j)
            xr_i = xr[:, None, :]
            xr_j = xr[None, :, :]
            xi_i = xi[:, None, :]
            xi_j = xi[None, :, :]
            real_acc += tl.sum(xr_i * xr_j + xi_i * xi_j, axis=2)
            imag_acc += tl.sum(xi_i * xr_j - xr_i * xi_j, axis=2)

        real_acc *= inv_P
        imag_acc *= inv_P

        lt0 = tl.arange(0, Lt)[:, None]
        lt1 = tl.arange(0, Lt)[None, :]
        base_o = pid_b * stride_o_b + lt0 * stride_o_lt0 + lt1 * stride_o_lt1
        tl.store(O_ptr + base_o + 0 * stride_o_c, real_acc)
        tl.store(O_ptr + base_o + 1 * stride_o_c, imag_acc)

    @triton.jit
    def _tyler_invgemm_weights_update_cov_kernel(
        Linv_ptr,  # fp32 view: (B, Lt, Lt, 2)   (lower-tri inverse)
        S_ptr,  # fp32 view: (B, Lt, P, 2)
        O_ptr,  # fp32 view: (B, Lt, Lt, 2)
        P,
        Lt: tl.constexpr,
        eps,
        w_max,
        inv_P,
        stride_l_b,
        stride_l_lt0,
        stride_l_lt1,
        stride_l_c,
        stride_s_b,
        stride_s_lt,
        stride_s_p,
        stride_s_c,
        stride_o_b,
        stride_o_lt0,
        stride_o_lt1,
        stride_o_c,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        lt = tl.arange(0, Lt)
        lt0 = lt[:, None]
        lt1 = lt[None, :]

        real_acc = tl.zeros((Lt, Lt), dtype=tl.float32)
        imag_acc = tl.zeros((Lt, Lt), dtype=tl.float32)

        for p0 in range(0, P, BLOCK_P):
            p = p0 + tl.arange(0, BLOCK_P)
            mask_p = p < P

            # Load S block (Lt,BLOCK_P)
            base_s = pid_b * stride_s_b + lt[:, None] * stride_s_lt + p[None, :] * stride_s_p
            sr = tl.load(
                S_ptr + base_s + 0 * stride_s_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            si = tl.load(
                S_ptr + base_s + 1 * stride_s_c,
                mask=mask_p[None, :],
                other=0.0,
            ).to(tl.float32)

            # q[p] = sum_i |(Linv @ S)_i|^2, computed row-by-row to avoid
            # materializing the full Y=(Linv@S) block.
            q = tl.zeros((BLOCK_P,), dtype=tl.float32)
            for i in range(0, Lt):
                base_l = pid_b * stride_l_b + i * stride_l_lt0 + lt * stride_l_lt1
                ar = tl.load(Linv_ptr + base_l + 0 * stride_l_c).to(tl.float32)  # (Lt,)
                ai = tl.load(Linv_ptr + base_l + 1 * stride_l_c).to(tl.float32)  # (Lt,)
                yr = tl.sum(ar[:, None] * sr - ai[:, None] * si, axis=0)
                yi = tl.sum(ar[:, None] * si + ai[:, None] * sr, axis=0)
                q += yr * yr + yi * yi

            q = tl.maximum(q, eps)
            w = (tl.full((BLOCK_P,), Lt, tl.float32) / q).to(tl.float32)
            w = tl.minimum(w, w_max)

            # Weight S in-place: Xw = S * w
            sr = sr * w[None, :]
            si = si * w[None, :]

            # Accumulate Xw @ Xw^H (complex):
            #   real = sum_p (xr_i*xr_j + xi_i*xi_j)
            #   imag = sum_p (xi_i*xr_j - xr_i*xi_j)
            xr_i = sr[:, None, :]
            xr_j = sr[None, :, :]
            xi_i = si[:, None, :]
            xi_j = si[None, :, :]
            real_acc += tl.sum(xr_i * xr_j + xi_i * xi_j, axis=2)
            imag_acc += tl.sum(xi_i * xr_j - xr_i * xi_j, axis=2)

        real_acc *= inv_P
        imag_acc *= inv_P

        base_o = pid_b * stride_o_b + lt0 * stride_o_lt0 + lt1 * stride_o_lt1
        tl.store(O_ptr + base_o + 0 * stride_o_c, real_acc)
        tl.store(O_ptr + base_o + 1 * stride_o_c, imag_acc)


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


def tyler_cov_update_triton(
    Xw_B_Lt_P: "torch.Tensor",
    out_B_Lt_Lt: "torch.Tensor",
    *,
    Lt: int,
    inv_P: float,
    cfg: TylerCovUpdateConfig = TylerCovUpdateConfig(),
):
    """
    Batched covariance update (complex64) using Triton:

      out[b] = inv_P * Xw[b] @ Xw[b]^H

    Notes
    -----
    - Xw has shape (B, Lt, P) complex64.
    - out has shape (B, Lt, Lt) complex64.
    - Implementation operates on the fp32 `view_as_real` storage.
    - Intended for small Lt (e.g. Lt=8/16) where cuBLAS batched GEMM can be launch/overhead bound.
    """
    import torch  # local import; torch is an unconditional repo dependency

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if not (Xw_B_Lt_P.is_cuda and out_B_Lt_Lt.is_cuda):
        raise ValueError("Triton Tyler cov update requires CUDA tensors.")
    if Xw_B_Lt_P.dtype != torch.complex64 or out_B_Lt_Lt.dtype != torch.complex64:
        raise ValueError("Triton Tyler cov update currently supports complex64 only.")
    if Xw_B_Lt_P.ndim != 3 or out_B_Lt_Lt.ndim != 3:
        raise ValueError("Expected Xw (B,Lt,P) and out (B,Lt,Lt).")
    B, Lt_in, P = (int(Xw_B_Lt_P.shape[0]), int(Xw_B_Lt_P.shape[1]), int(Xw_B_Lt_P.shape[2]))
    if int(Lt) != int(Lt_in):
        raise ValueError(f"Lt mismatch: arg Lt={int(Lt)} tensor Lt={int(Lt_in)}")
    if out_B_Lt_Lt.shape[0] != B or out_B_Lt_Lt.shape[1] != Lt_in or out_B_Lt_Lt.shape[2] != Lt_in:
        raise ValueError("out must have shape (B,Lt,Lt).")
    if int(Lt) > 16:
        raise ValueError("Triton Tyler cov update supports Lt <= 16 (fallback to torch for larger Lt).")

    X = torch.view_as_real(Xw_B_Lt_P)  # (B,Lt,P,2) float32 view
    O = torch.view_as_real(out_B_Lt_Lt)  # (B,Lt,Lt,2) float32 view

    sxb, sxlt, sxp, sxc = (int(s) for s in X.stride())
    sob, solt0, solt1, soc = (int(s) for s in O.stride())

    grid = (B,)
    _tyler_cov_update_kernel[grid](
        X,
        O,
        P=P,
        Lt=int(Lt),
        inv_P=float(inv_P),
        stride_x_b=sxb,
        stride_x_lt=sxlt,
        stride_x_p=sxp,
        stride_x_c=sxc,
        stride_o_b=sob,
        stride_o_lt0=solt0,
        stride_o_lt1=solt1,
        stride_o_c=soc,
        BLOCK_P=int(cfg.block_p),
        num_warps=int(cfg.num_warps),
    )


def tyler_weights_update_cov_triton(
    Y_B_Lt_P: "torch.Tensor",
    S_B_Lt_P: "torch.Tensor",
    out_B_Lt_Lt: "torch.Tensor",
    *,
    Lt: int,
    eps: float,
    inv_P: float,
    cfg: TylerWeightsUpdateConfig = TylerWeightsUpdateConfig(),
):
    """
    Fused Tyler weights + covariance update (complex64) using Triton:

      q[b,p] = sum_lt |Y[b,lt,p]|^2
      w[b,p] = min(Lt / max(q, eps), w_max)
      out[b] = inv_P * (S[b] * w) @ (S[b] * w)^H

    Notes
    -----
    - Y/S have shape (B, Lt, P) complex64.
    - out has shape (B, Lt, Lt) complex64.
    - Implementation operates on the fp32 `view_as_real` storage.
    - Intended for small Lt (e.g. Lt=8/16) to reduce kernel launches and
      avoid materializing Xw in global memory.
    """
    import torch  # local import; torch is an unconditional repo dependency

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if not (Y_B_Lt_P.is_cuda and S_B_Lt_P.is_cuda and out_B_Lt_Lt.is_cuda):
        raise ValueError("Triton Tyler fused update requires CUDA tensors.")
    if (
        Y_B_Lt_P.dtype != torch.complex64
        or S_B_Lt_P.dtype != torch.complex64
        or out_B_Lt_Lt.dtype != torch.complex64
    ):
        raise ValueError("Triton Tyler fused update currently supports complex64 only.")
    if Y_B_Lt_P.shape != S_B_Lt_P.shape:
        raise ValueError("Y and S must have identical shapes (B,Lt,P).")
    if Y_B_Lt_P.ndim != 3:
        raise ValueError(f"Expected (B,Lt,P), got {tuple(Y_B_Lt_P.shape)}")
    B, Lt_in, P = (int(Y_B_Lt_P.shape[0]), int(Y_B_Lt_P.shape[1]), int(Y_B_Lt_P.shape[2]))
    if int(Lt) != int(Lt_in):
        raise ValueError(f"Lt mismatch: arg Lt={int(Lt)} tensor Lt={int(Lt_in)}")
    if out_B_Lt_Lt.shape[0] != B or out_B_Lt_Lt.shape[1] != Lt_in or out_B_Lt_Lt.shape[2] != Lt_in:
        raise ValueError("out must have shape (B,Lt,Lt).")
    if int(Lt) > 16:
        raise ValueError("Triton Tyler fused update supports Lt <= 16 (fallback to torch for larger Lt).")

    Y = torch.view_as_real(Y_B_Lt_P)  # (B,Lt,P,2) float32 view
    S = torch.view_as_real(S_B_Lt_P)
    O = torch.view_as_real(out_B_Lt_Lt)  # (B,Lt,Lt,2) float32 view

    syb, sylt, syp, syc = (int(s) for s in Y.stride())
    ssb, sslt, ssp, ssc = (int(s) for s in S.stride())
    sob, solt0, solt1, soc = (int(s) for s in O.stride())

    grid = (B,)
    _tyler_weights_update_cov_kernel[grid](
        Y,
        S,
        O,
        P=P,
        Lt=int(Lt),
        eps=float(eps),
        w_max=float(cfg.w_max),
        inv_P=float(inv_P),
        stride_y_b=syb,
        stride_y_lt=sylt,
        stride_y_p=syp,
        stride_y_c=syc,
        stride_s_b=ssb,
        stride_s_lt=sslt,
        stride_s_p=ssp,
        stride_s_c=ssc,
        stride_o_b=sob,
        stride_o_lt0=solt0,
        stride_o_lt1=solt1,
        stride_o_c=soc,
        BLOCK_P=int(cfg.block_p),
        num_warps=int(cfg.num_warps),
    )


def tyler_invgemm_weights_update_cov_triton(
    Linv_B_Lt_Lt: "torch.Tensor",
    S_B_Lt_P: "torch.Tensor",
    out_B_Lt_Lt: "torch.Tensor",
    *,
    Lt: int,
    eps: float,
    inv_P: float,
    cfg: TylerInvGemmUpdateConfig = TylerInvGemmUpdateConfig(),
):
    """
    Fused Tyler (inv-gemm solve) + weights + covariance update (complex64) using Triton:

      q[b,p] = sum_i |(L^{-1} @ S)[b,i,p]|^2
      w[b,p] = min(Lt / max(q, eps), w_max)
      out[b] = inv_P * (S[b] * w) @ (S[b] * w)^H

    Notes
    -----
    - Linv has shape (B, Lt, Lt) complex64 (lower-triangular inverse).
    - S has shape (B, Lt, P) complex64.
    - out has shape (B, Lt, Lt) complex64.
    - Intended for small Lt (e.g. Lt=8/16) when STAP_TYLER_SOLVE_MODE=inv_gemm.
    - Avoids materializing Y=(Linv@S) in global memory.
    """
    import torch  # local import; torch is an unconditional repo dependency

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if not (Linv_B_Lt_Lt.is_cuda and S_B_Lt_P.is_cuda and out_B_Lt_Lt.is_cuda):
        raise ValueError("Triton Tyler inv-gemm fused update requires CUDA tensors.")
    if (
        Linv_B_Lt_Lt.dtype != torch.complex64
        or S_B_Lt_P.dtype != torch.complex64
        or out_B_Lt_Lt.dtype != torch.complex64
    ):
        raise ValueError("Triton Tyler inv-gemm fused update currently supports complex64 only.")
    if Linv_B_Lt_Lt.ndim != 3 or S_B_Lt_P.ndim != 3 or out_B_Lt_Lt.ndim != 3:
        raise ValueError("Expected Linv (B,Lt,Lt), S (B,Lt,P), out (B,Lt,Lt).")

    B, Lt_in, Lt_in2 = (
        int(Linv_B_Lt_Lt.shape[0]),
        int(Linv_B_Lt_Lt.shape[1]),
        int(Linv_B_Lt_Lt.shape[2]),
    )
    if int(Lt_in) != int(Lt_in2):
        raise ValueError("Linv must have shape (B,Lt,Lt).")
    if int(Lt) != int(Lt_in):
        raise ValueError(f"Lt mismatch: arg Lt={int(Lt)} tensor Lt={int(Lt_in)}")
    if S_B_Lt_P.shape[0] != B or int(S_B_Lt_P.shape[1]) != int(Lt_in):
        raise ValueError("S must have shape (B,Lt,P).")
    P = int(S_B_Lt_P.shape[2])
    if out_B_Lt_Lt.shape[0] != B or out_B_Lt_Lt.shape[1] != Lt_in or out_B_Lt_Lt.shape[2] != Lt_in:
        raise ValueError("out must have shape (B,Lt,Lt).")
    if int(Lt) > 16:
        raise ValueError("Triton Tyler inv-gemm fused update supports Lt <= 16 (fallback to torch for larger Lt).")

    Linv = torch.view_as_real(Linv_B_Lt_Lt)  # (B,Lt,Lt,2) float32 view
    S = torch.view_as_real(S_B_Lt_P)  # (B,Lt,P,2) float32 view
    O = torch.view_as_real(out_B_Lt_Lt)  # (B,Lt,Lt,2) float32 view

    slb, sllt0, sllt1, slc = (int(s) for s in Linv.stride())
    ssb, sslt, ssp, ssc = (int(s) for s in S.stride())
    sob, solt0, solt1, soc = (int(s) for s in O.stride())

    grid = (B,)
    _tyler_invgemm_weights_update_cov_kernel[grid](
        Linv,
        S,
        O,
        P=P,
        Lt=int(Lt),
        eps=float(eps),
        w_max=float(cfg.w_max),
        inv_P=float(inv_P),
        stride_l_b=slb,
        stride_l_lt0=sllt0,
        stride_l_lt1=sllt1,
        stride_l_c=slc,
        stride_s_b=ssb,
        stride_s_lt=sslt,
        stride_s_p=ssp,
        stride_s_c=ssc,
        stride_o_b=sob,
        stride_o_lt0=solt0,
        stride_o_lt1=solt1,
        stride_o_c=soc,
        BLOCK_P=int(cfg.block_p),
        num_warps=int(cfg.num_warps),
    )
