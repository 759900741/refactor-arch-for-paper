# -*- coding: utf-8 -*-
"""
metaheuristics.py
=================
面向超参搜索的元启发式算法集合：RIME / TE-RIME / PSO / BWO / GWO。

统一约定：
    - 算法内部只处理 u ∈ [0,1]^D 的连续向量，解码/修复由外部 decode(u)->cfg 完成；
    - 所有算法共用同一初始种群（同 seed）、同一评估函数、同一评估预算；
    - 每个算法使用独立的 np.random.RandomState，不触碰全局随机状态。

对外接口：
    run(algo, dim, fitness, pop=5, iters=100, seed=42, budget=None, patience=None) -> dict
"""
from __future__ import annotations
import time
import numpy as np


# ---------------------------------------------------------------------------
# 边界处理：越界均匀重初始化（TE-RIME 增强③） vs 截断
# ---------------------------------------------------------------------------
def _clip(u):
    return np.clip(u, 0.0, 1.0)


def _reinit_out_of_bounds(u, rng, lo=0.0, hi=1.0):
    """越界维度用均匀重初始化替代简单截断，维持种群多样性。"""
    v = np.array(u, dtype=float, copy=True)
    bad = (v < lo) | (v > hi)
    if bad.any():
        v[bad] = rng.uniform(lo, hi, bad.sum())
    return v


def _crossover_binomial(target, donor, rng, cr=0.9):
    """二项式交叉：至少保留一维来自 donor。"""
    d = target.size
    mask = rng.random(d) < cr
    if not mask.any():
        mask[rng.randint(d)] = True
    out = target.copy()
    out[mask] = donor[mask]
    return out


# ---------------------------------------------------------------------------
# 统一运行框架
# ---------------------------------------------------------------------------
def run(algo, dim, fitness, pop=5, iters=100, seed=42, budget=None, patience=None,
        log=print, x0=None):
    """执行一个算法并返回轨迹。

    algo    : "rime" | "terime" | "pso" | "bwo" | "gwo"
    dim     : 决策向量维数
    fitness : f(u: np.ndarray) -> float（越小越好），内部负责缓存与训练
    budget  : 唯一评估次数上限（None 则不限制，按 pop*iters 计）
    x0      : 可选初始种群 (k, dim)；行数不足 pop 时随机补齐，超出截断。
              仅用于 TE-RIME 长跑的 warm-start，表 5 等预算对比应传 None 保公平。
    """
    rng = np.random.RandomState(seed)
    algo = algo.lower()
    budget = int(budget or pop * iters)
    t0 = time.perf_counter()

    # 统一初始种群（所有算法共享，保证公平）；x0 提供时作为 warm-start
    if x0 is not None:
        X = np.asarray(x0, dtype=float)[:pop]
        if len(X) < pop:
            X = np.vstack([X, rng.uniform(0.0, 1.0, (pop - len(X), dim))])
    else:
        X = rng.uniform(0.0, 1.0, (pop, dim))
    fit = np.array([fitness(x) for x in X])
    best_i = int(np.argmin(fit))
    best, best_fit = X[best_i].copy(), float(fit[best_i])
    evals = pop
    hist = [dict(it=0, evals=evals, best_fit=best_fit, mean_fit=float(fit.mean()))]

    # PSO/BWO 状态
    vel = rng.uniform(-0.1, 0.1, (pop, dim))
    pbest, pbest_fit = X.copy(), fit.copy()

    stall = 0
    for it in range(1, iters + 1):
        if evals >= budget:
            break
        prev_best = best_fit
        New = X.copy()

        if algo in ("rime", "terime"):
            h = 1.0 - it / iters
            beta = max(0.0, 1.0 - 5.0 * it / iters)     # 步长系数 W=5
            E = np.sqrt(it / iters)                     # 软/硬雾凇切换阈值
            for i in range(pop):
                r1 = rng.uniform(-1, 1); theta = rng.uniform(0, 2 * np.pi)
                if rng.random() < E:                    # 软雾凇：全局搜索
                    v = best + r1 * np.cos(theta) * beta * (h * (1.0 - 0.0) + 0.0)
                else:                                   # 硬雾凇：向最优穿刺
                    r2 = rng.uniform(-1, 1)
                    v = best + r2 * (best - X[i])
                if algo == "terime":
                    # 增强①：DE/rand/1 变异（提升探索）
                    if rng.random() < 0.5 and pop >= 4:
                        a, b, c = rng.choice(pop, 3, replace=False)
                        F = rng.uniform(0.5, 0.8)
                        v = _crossover_binomial(v, X[a] + F * (X[b] - X[c]), rng, 0.9)
                    # 增强②：高斯邻域 / 交叉 双路径开发
                    if rng.random() < 0.5:
                        sigma = 0.1 * (1.0 - it / iters)
                        v = _crossover_binomial(v, best + sigma * rng.randn(dim), rng, 0.9)
                    else:
                        C = 1.0 - it / iters
                        v = _crossover_binomial(v, v, rng, C)
                    # 增强③：越界均匀重初始化
                    v = _reinit_out_of_bounds(v, rng)
                else:
                    v = _clip(v)
                New[i] = v

        elif algo == "pso":
            w, c1, c2 = 0.7298, 1.4962, 1.4962
            for i in range(pop):
                r1, r2 = rng.random(dim), rng.random(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (best - X[i])
                vel[i] = np.clip(vel[i], -0.3, 0.3)
                New[i] = _clip(X[i] + vel[i])

        elif algo == "bwo":
            # 白鲸优化：游泳（探索）/ 猎物（开发）/ 鲸落（跳出）
            Wf = 0.1 - 0.05 * it / iters                # 鲸落概率随迭代下降
            Bf = 2.0 * (1.0 - it / iters)               # 平衡因子
            for i in range(pop):
                if rng.random() < 0.5:
                    r = rng.random()
                    a = 2.0 * r - 1.0
                    C = 2.0 * r
                    L = (Wf - 1.0) * rng.random() + 1.0
                    Move = np.exp(-a * it / iters) * np.cos(2 * np.pi * a) * L
                    New[i] = _clip(X[i] + Bf * Move * (1.0 - X[i]))
                else:
                    u1, u2 = rng.choice(pop, 2, replace=False)
                    New[i] = _clip(X[i] + Bf * (X[u1] - X[u2]))
                if Bf * rng.random() < Wf:              # 鲸落：随机重定位
                    New[i] = rng.uniform(0.0, 1.0, dim)
        elif algo == "gwo":
            a = 2.0 - 2.0 * it / iters
            order = np.argsort(fit)
            Xa, Xb, Xc = X[order[0]], X[order[min(1, pop - 1)]], X[order[min(2, pop - 1)]]
            for i in range(pop):
                X1 = Xa - a * np.abs(rng.random(dim) * Xa - X[i])
                X2 = Xb - a * np.abs(rng.random(dim) * Xb - X[i])
                X3 = Xc - a * np.abs(rng.random(dim) * Xc - X[i])
                New[i] = _clip((X1 + X2 + X3) / 3.0)
        else:
            raise ValueError(f"未知算法: {algo}")

        # 评估新种群（评估预算按调用次数计）
        for i in range(pop):
            if evals >= budget:
                New[i] = X[i]; fit[i] = fit[i]
                continue
            f_new = fitness(New[i])
            evals += 1
            if f_new < fit[i]:                       # 正向贪婪选择
                X[i] = New[i]; fit[i] = f_new
                if f_new < best_fit:
                    best, best_fit = New[i].copy(), float(f_new)
            # PSO/BWO 个体历史最优
            if f_new < pbest_fit[i]:
                pbest[i], pbest_fit[i] = New[i].copy(), f_new
        hist.append(dict(it=it, evals=evals, best_fit=best_fit, mean_fit=float(fit.mean())))

        stall = stall + 1 if best_fit >= prev_best - 1e-12 else 0
        if patience and stall >= patience:
            log(f"    [{algo}] 停在 it={it}（连续 {patience} 代无改进）")
            break

    return dict(algo=algo, best_u=best, best_fit=float(best_fit), history=hist,
                evals=int(evals), wall_sec=round(time.perf_counter() - t0, 2), seed=seed,
                pop=pop, iters=iters, budget=budget)


if __name__ == "__main__":
    # 自检：在 Rastrigin 上比较收敛
    def fit(u):
        x = u * 10.24 - 5.12
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    for a in ("rime", "terime", "pso", "bwo", "gwo"):
        r = run(a, 6, fit, pop=5, iters=60, seed=1, log=lambda *x, **k: None)
        print(f"{a:8s} best={r['best_fit']:.4f} evals={r['evals']} {r['wall_sec']}s")
