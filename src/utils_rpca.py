# -*- coding: utf-8 -*-
"""
utils_rpca.py
=============
RPCA（鲁棒主成分分析）工具库。

核心：
- rpca_ialm      : 增广拉格朗日乘子法解 min ||L||_* + lam||S||_1 s.t. X=L+S
- svd_select_k   : 对 L 做 SVD，按累积能量(默认95%)选主成分数 k，返回右奇异向量 V_k
- project_denoise: 用 P = V_k V_k^T 做线性投影去噪 L = X P, S = X - L
- 辅助           : SVT / 软阈值 / 符号固定

注意：V_k / P 必须仅由训练集估计后作用于验证/测试，防止跨集信息泄漏。
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# 基础算子
# ----------------------------------------------------------------------------
def svt(M: np.ndarray, tau: float):
    """奇异值软阈值(Singular Value Thresholding)。返回 (阈值化矩阵, 原奇异值)。"""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s2 = np.maximum(s - tau, 0.0)
    return (U * s2) @ Vt, s


def shrink(M: np.ndarray, tau: float) -> np.ndarray:
    """逐元素软阈值。"""
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)


# ----------------------------------------------------------------------------
# RPCA - IALM
# ----------------------------------------------------------------------------
def rpca_ialm(X: np.ndarray, lam: float, tol: float = 1e-6, max_iter: int = 500,
              rho: float = 1.5, mu_max: float = 1e10):
    """
    解 min ||L||_* + lam * ||S||_1,  s.t. X = L + S
    返回 (L, S, info)。info: iters / converged / recon_rel_err。
    """
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    normX = np.linalg.norm(X, "fro") + 1e-12
    norm2 = np.linalg.norm(X, 2) + 1e-12
    norm_inf = np.linalg.norm(X, np.inf) + 1e-12

    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = X / max(norm2, norm_inf / max(lam, 1e-12))
    mu = 1.25 / norm2

    converged = False
    it = 0
    err = 1.0
    for it in range(1, max_iter + 1):
        L, _ = svt(X - S + Y / mu, 1.0 / mu)
        S = shrink(X - L + Y / mu, lam / mu)
        Z = X - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_max)
        err = np.linalg.norm(Z, "fro") / normX
        if err < tol:
            converged = True
            break
    info = dict(iters=int(it), converged=bool(converged), recon_rel_err=float(err))
    return L, S, info


# ----------------------------------------------------------------------------
# SVD 选主成分
# ----------------------------------------------------------------------------
def svd_select_k(L: np.ndarray, energy: float = 0.95):
    """
    对 L 做 SVD，按累积能量阈值选 k，返回 dict：
        k, singular_values, ratio, cum, V (F x k, 已符号固定)
    """
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    s2 = s ** 2
    ratio = s2 / (s2.sum() + 1e-12)
    cum = np.cumsum(ratio)
    k = int(np.searchsorted(cum, energy) + 1)
    k = max(1, min(k, len(s)))
    V = Vt[:k].T.copy()  # (F, k)
    # 符号固定：每列绝对值最大分量为正，保证可复现
    for j in range(V.shape[1]):
        idx = int(np.argmax(np.abs(V[:, j])))
        if V[idx, j] < 0:
            V[:, j] *= -1.0
    return dict(k=k, singular_values=s, ratio=ratio, cum=cum, V=V)


# ----------------------------------------------------------------------------
# 投影去噪
# ----------------------------------------------------------------------------
def project_denoise(X: np.ndarray, V: np.ndarray):
    """用训练集子空间 V(F x k) 投影去噪：P=V V^T, L=X P, S=X-L。"""
    P = V @ V.T
    L = X @ P
    S = X - L
    return L, S, P
