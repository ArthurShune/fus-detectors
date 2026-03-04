"""Replay CUDA warmup helpers."""

from __future__ import annotations

import argparse
import os

import numpy as np

from sim.kwave.common import SimGeom


def run_cuda_warmup(args: argparse.Namespace) -> None:
    if not getattr(args, "cuda_warmup", True):
        return
    dev = str(getattr(args, "stap_device", "cuda") or "")
    if not dev.lower().startswith("cuda"):
        return
    try:
        import torch  # local import (torch optional)

        if torch.cuda.is_available():
            _ = torch.empty((1,), device=dev)
            a = torch.randn((32, 32), device=dev, dtype=torch.float32)
            b = torch.randn((32, 32), device=dev, dtype=torch.float32)
            ac = (a + 1j * b).to(dtype=torch.complex64)
            bc = (b + 1j * a).to(dtype=torch.complex64)
            _ = ac @ bc
            c = ac @ ac.conj().T
            _ = torch.linalg.eigh(c)
            c_pd = c + (1e-2 * torch.eye(c.shape[0], device=dev, dtype=c.dtype))
            chol = torch.linalg.cholesky(c_pd)
            rr = torch.randn((c.shape[0], 64), device=dev, dtype=torch.float32)
            ri = torch.randn((c.shape[0], 64), device=dev, dtype=torch.float32)
            rhs = (rr + 1j * ri).to(dtype=c.dtype)
            _ = torch.linalg.solve_triangular(chol, rhs, upper=False)
            _ = torch.cholesky_solve(rhs, chol)
            x = torch.randn((32, 32), device=dev, dtype=torch.float32)
            xc = (x + 1j * x).to(dtype=torch.complex64)
            _ = torch.fft.fft2(xc)
            v = torch.rand((1024,), device=dev, dtype=torch.float32)
            _ = torch.sort(v)
            _ = torch.topk(v, k=32)
            torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - optional warmup
        print(f"[replay_stap_from_run] CUDA warmup skipped: {exc}")


def run_heavy_cuda_warmup(args: argparse.Namespace, *, geom: SimGeom) -> None:
    if os.getenv("CUDA_WARMUP_HEAVY", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    dev = str(getattr(args, "stap_device", "cuda") or "")
    if not dev.lower().startswith("cuda"):
        return
    try:
        import torch  # local import (torch optional)

        if torch.cuda.is_available():
            env_mcsvd_torch = os.getenv("MC_SVD_TORCH", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            env_reg_torch = os.getenv("MC_SVD_REG_TORCH", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            baseline_type = str(getattr(args, "baseline", "mc_svd") or "").strip().lower()
            if (
                env_mcsvd_torch
                and env_reg_torch
                and baseline_type in {"mc_svd", "svd"}
                and bool(getattr(args, "reg_enable", False))
            ):
                from sim.kwave.common import _baseline_pd_mcsvd

                t_warm = int(getattr(args, "time_window_length", 0) or 32)
                h_warm = int(getattr(geom, "Ny", 0) or 0)
                w_warm = int(getattr(geom, "Nx", 0) or 0)
                if t_warm > 1 and h_warm > 0 and w_warm > 0:
                    print(
                        f"[replay_stap_from_run] Heavy CUDA warmup (MC-SVD torch reg): "
                        f"T={t_warm}, H={h_warm}, W={w_warm}",
                        flush=True,
                    )
                    rng = np.random.default_rng(0)
                    dummy = (
                        rng.standard_normal((t_warm, h_warm, w_warm), dtype=np.float32)
                        + 1j * rng.standard_normal((t_warm, h_warm, w_warm), dtype=np.float32)
                    ).astype(np.complex64, copy=False)
                    _baseline_pd_mcsvd(
                        dummy,
                        reg_enable=True,
                        reg_method=str(getattr(args, "reg_method", "phasecorr")),
                        reg_subpixel=int(getattr(args, "reg_subpixel", 4)),
                        reg_reference=str(getattr(args, "reg_reference", "median")),
                        svd_rank=getattr(args, "svd_rank", None),
                        svd_energy_frac=getattr(args, "svd_energy_frac", None),
                        device=dev,
                        return_filtered_cube=False,
                    )

            env_fast = os.getenv("STAP_FAST_PATH", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if env_fast:
                lt_warm = int(getattr(args, "lt", 0) or 0)
                t_warm = int(getattr(args, "time_window_length", 0) or 0)
                th_warm = int(getattr(args, "tile_h", 0) or 0)
                tw_warm = int(getattr(args, "tile_w", 0) or 0)
                if t_warm <= 0:
                    t_warm = 32
                if lt_warm <= 0:
                    lt_warm = min(8, max(2, t_warm - 1))
                if th_warm <= 0:
                    th_warm = 8
                if tw_warm <= 0:
                    tw_warm = 8
                if 2 <= lt_warm < t_warm and th_warm > 0 and tw_warm > 0:
                    stride_env = os.getenv("STAP_SNAPSHOT_STRIDE", "").strip()
                    max_env = os.getenv("STAP_MAX_SNAPSHOTS", "").strip()
                    try:
                        stride = int(stride_env) if stride_env else 1
                    except ValueError:
                        stride = 1
                    stride = max(1, stride)
                    try:
                        max_snaps = int(max_env) if max_env else None
                    except ValueError:
                        max_snaps = None

                    n_full = t_warm - lt_warm + 1
                    n_eff = (n_full + stride - 1) // stride
                    if max_snaps is not None and max_snaps > 0 and n_eff > max_snaps:
                        n_eff = int(max_snaps)
                    p = int(max(1, n_eff) * th_warm * tw_warm)

                    b_target = 192
                    p_target = int(max(1, p))
                    max_bytes = 64 * 1024 * 1024
                    denom = int(max(1, lt_warm) * max(1, p_target))
                    max_b = int(max(1, (max_bytes // 16) // denom))
                    b_warm = int(min(b_target, max_b))
                    p_warm = p_target
                    if b_warm < 8:
                        b_warm = 8
                        max_p = int(max(1, (max_bytes // 16) // (b_warm * max(1, lt_warm))))
                        p_warm = int(min(p_target, max_p))
                    if p_warm <= 0:
                        p_warm = 1
                    try:
                        k_warm = int(getattr(args, "max_pts", 15) or 15)
                    except Exception:
                        k_warm = 15
                    k_warm = max(3, min(k_warm, 21))
                    print(
                        f"[replay_stap_from_run] Heavy CUDA warmup (STAP core): "
                        f"B={b_warm}, Lt={lt_warm}, P={p_warm}, K={k_warm}",
                        flush=True,
                    )

                    rr = torch.randn((b_warm, lt_warm, p_warm), device=dev, dtype=torch.float32)
                    ri = torch.randn((b_warm, lt_warm, p_warm), device=dev, dtype=torch.float32)
                    s = (rr + 1j * ri).to(dtype=torch.complex64)
                    r = torch.matmul(s, s.conj().transpose(-2, -1)) / float(p_warm)
                    herm = 0.5 * (r + r.conj().transpose(-2, -1))
                    _ = torch.linalg.eigvalsh(herm).real

                    eye = torch.eye(lt_warm, device=dev, dtype=herm.dtype).unsqueeze(0)
                    r_lam = herm + (1e-2 * eye)
                    chol = torch.linalg.cholesky(r_lam)
                    y = torch.linalg.solve_triangular(chol, s, upper=False)

                    triton_env = os.getenv("STAP_TYLER_TRITON_WEIGHTS", "").strip().lower()
                    if triton_env in {"1", "true", "yes", "on"}:
                        warm_triton = True
                    elif triton_env in {"0", "false", "no", "off"}:
                        warm_triton = False
                    else:
                        warm_triton = True
                    if warm_triton:
                        try:
                            from pipeline.stap.triton_ops import (
                                TylerWeightsConfig,
                                tyler_weights_scale_triton,
                            )
                            from pipeline.stap.triton_ops import (
                                triton_available as _triton_available,
                            )

                            if _triton_available():
                                out = torch.empty_like(s)
                                tyler_weights_scale_triton(
                                    y,
                                    s,
                                    out,
                                    Lt=int(lt_warm),
                                    eps=1e-8,
                                    cfg=TylerWeightsConfig(),
                                )
                        except Exception:
                            pass

                    cr = torch.randn((b_warm, lt_warm, k_warm), device=dev, dtype=torch.float32)
                    ci = torch.randn((b_warm, lt_warm, k_warm), device=dev, dtype=torch.float32)
                    ct_exp = (cr + 1j * ci).to(dtype=torch.complex64)
                    cw = torch.linalg.solve_triangular(chol, ct_exp, upper=False)
                    gram = torch.bmm(cw.conj().transpose(1, 2), cw)
                    eye_k = torch.eye(k_warm, device=dev, dtype=gram.dtype).unsqueeze(0)
                    gram = gram + (1e-2 * eye_k)
                    lg = torch.linalg.cholesky(gram)

                    ar = torch.randn((b_warm, k_warm, k_warm), device=dev, dtype=torch.float32)
                    ai = torch.randn((b_warm, k_warm, k_warm), device=dev, dtype=torch.float32)
                    a = (ar + 1j * ai).to(dtype=torch.complex64)
                    _ = torch.cholesky_solve(a, lg)
                    torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - optional warmup
        print(f"[replay_stap_from_run] Heavy CUDA warmup skipped: {exc}", flush=True)
