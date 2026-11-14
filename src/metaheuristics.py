# -*- coding: utf-8 -*-
"""
metaheuristics.py
=================
面向超参搜索的元启发式算法集合：RIME / TE-RIME / PSO / BWO / GWO。

统一约定：
    - 算法内部只处理 u ∈ [0,1]^D 的连续向量，解码/修复由外部 decode(u)->cfg 完成；
    - 所有算法共用同一初始种群（同 seed）、同一评估函数、同一评估预算；
    - 每个算法使用独立的 np.random.RandomState，不触碰全局随机状态。

v4（stage5-tev4）相对 v3 的改动：
    S2 调度去退化     —— 所有调度量改用「评估进度 p = evals/budget」，RIME 的 β 回落原文
                         2(1-p) 形式并设地板，修掉「t>0.2T 后 β=0、软雾凇退化为复制 best」。
    S3 边界反射       —— TE-RIME 越界维改为反射（v←2b−v）后截断，替代随机重采样；
                         最优点贴在箱体边界时不再被弹回内部。
    S4 自适应算子池   —— TE-RIME 三类提案族（RIME 核心 / DE 变异 / 精英高斯-QOBL）按成功率
                         做概率匹配；DE 的 F/CR 用 SHADE 成功记忆（Lehmer 均值）。
    S5 坐标精修       —— TE-RIME 每代预留探测槽，对精英做 ±1 格点坐标搜索（Hooke-Jeeves
                         型，含 pattern move）；stall≥2 时改为全维扫。只更新 best，不污染种群。
    S6 准对立重启     —— stall≥restart_after 时用 QOBL 生成个体替换最差，计入预算。
    S7 基线校订       —— GWO 补回 A=2a·r1−a、C=2·r2；BWO 按 Zhong et al. 2022 原式重写
                         （Bf 平衡因子、sin/cos 游泳与镜像、C1·LF 捕食项、鲸落 X_step）。

对外接口：
    run(algo, dim, fitness, pop=5, iters=100, seed=42, budget=None, patience=None,
        log=print, x0=None, identity=None, restart_after=None, grid=None, refine=True)
"""
from __future__ import annotations
import math
import time
import numpy as np

ALGORITHM_VERSION = "stage5-tev4"


# ---------------------------------------------------------------------------
# 边界处理
# ---------------------------------------------------------------------------
def _clip(u):
    return np.clip(u, 0.0, 1.0)


def _reflect(u, lo=0.0, hi=1.0, rounds=2):
    """S3：越界维反射 v ← 2b − v，仍越界（极端值）再截断。

    相比「越界即随机重采样」，反射保留了个体已获得的贴边结构：
    本问题的 fast 最优解 7/9 维顶在箱体边界，随机重采样会把它打回内部。
    """
    v = np.array(u, dtype=float, copy=True)
    v[~np.isfinite(v)] = 0.5
    for _ in range(rounds):
        bad_lo = v < lo
        bad_hi = v > hi
        if not (bad_lo.any() or bad_hi.any()):
            break
        v[bad_lo] = 2.0 * lo - v[bad_lo]
        v[bad_hi] = 2.0 * hi - v[bad_hi]
    return np.clip(v, lo, hi)


def _reinit_out_of_bounds(u, rng, lo=0.0, hi=1.0):
    """旧版越界均匀重初始化（保留供对照，v4 的 TE-RIME 默认不再使用）。"""
    v = np.array(u, dtype=float, copy=True)
    bad = (v < lo) | (v > hi)
    if bad.any():
        v[bad] = rng.uniform(lo, hi, bad.sum())
    return v


def _qobl(x, rng):
    """S6：准对立学习，以搜索区间中心 0.5 为对称中心的反射解。"""
    c = 0.5
    v = c + rng.random(np.asarray(x).size) * (c - np.asarray(x))
    return np.clip(v, 0.0, 1.0)


def _crossover_binomial(target, donor, rng, cr=0.9):
    """二项式交叉：至少保留一维来自 donor。"""
    d = target.size
    mask = rng.random(d) < cr
    if not mask.any():
        mask[rng.randint(d)] = True
    out = target.copy()
    out[mask] = donor[mask]
    return out


def _sample_F(rng, m_f=0.5):
    """SHADE：F ~ |Cauchy(m_f, 0.1)|，截断到 [0.05, 1]。"""
    f = m_f + 0.1 * rng.standard_cauchy()
    return float(np.clip(abs(f), 0.05, 1.0))


def _sample_CR(rng, m_cr=0.5):
    """SHADE：CR ~ N(m_cr, 0.1)，截断到 [0, 1]。"""
    return float(np.clip(m_cr + 0.1 * rng.randn(), 0.0, 1.0))


def _lerp_mean(vals):
    """Lehmer 均值（SHADE 记忆更新）。"""
    v = np.asarray(vals, dtype=float)
    den = float(v.sum())
    return float((v * v).sum() / den) if den > 1e-12 else 0.5


def _levy(dim, rng, beta=1.5):
    """BWO 开发阶段使用的 Lévy 飞行（Zhong et al. 2022）。"""
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (num / den) ** (1.0 / beta)
    u = rng.randn(dim) * sigma
    v = np.abs(rng.randn(dim))
    return 0.05 * u / np.maximum(v ** (1.0 / beta), 1e-12)


def _grid_step(u, j, n, k):
    """S5：在第 j 维沿格点索引走 k 步（编码到格心 (i+0.5)/n）。越界返回 None。"""
    i = int(np.floor(np.clip(u[j], 0.0, 0.999999) * n))
    i2 = i + k
    if i2 < 0 or i2 > n - 1:
        return None
    v = np.array(u, dtype=float, copy=True)
    v[j] = (i2 + 0.5) / n
    return v


class _OpPool:
    """S4：概率匹配式算子选择（指数遗忘计数，带概率下限）。"""

    NAMES = ("rime", "de", "elite")

    def __init__(self, n=3, pmin=0.1, decay=0.9, prior=(1.5, 1.0, 1.0)):
        # 以 RIME 核心为先验（本问题的硬雾凇穿刺天然朝箱体边界饱和），
        # 避免自适应在小样本早期因 1~2 次偶然失败就把主算子判死。
        self.succ = np.array(prior[:n], dtype=float)
        self.trial = np.full(n, 2.5)
        self.pmin = pmin
        self.decay = decay

    def probs(self):
        p = self.succ / self.trial
        p = np.maximum(p, 1e-6)
        p = p / p.sum()
        p = np.maximum(p, self.pmin)
        return p / p.sum()

    def select(self, rng):
        return int(rng.choice(len(self.succ), p=self.probs()))

    def update(self, k, success):
        self.trial[k] = self.decay * self.trial[k] + 1.0
        self.succ[k] = self.decay * self.succ[k] + (1.0 if success else 0.0)

    def as_dict(self):
        p = self.probs()
        return {n: round(float(p[i]), 4) for i, n in enumerate(self.NAMES)}


# ---------------------------------------------------------------------------
# 统一运行框架
# ---------------------------------------------------------------------------
def run(algo, dim, fitness, pop=5, iters=100, seed=42, budget=None, patience=None,
        log=print, x0=None, identity=None, restart_after=None, grid=None, refine=True):
    """执行一个算法并返回轨迹。

    algo    : "rime" | "terime" | "pso" | "bwo" | "gwo"
    dim     : 决策向量维数
    fitness : f(u: np.ndarray) -> float（越小越好），内部负责缓存与训练
    budget  : 评估上限，默认 pop*iters；传 identity 时按独立配置计数。
    x0      : 可选初始种群 (k, dim)；行数不足 pop 时随机补齐，超出截断。
              仅用于 TE-RIME 长跑的 warm-start，表 5 等预算对比应传 None 保公平。
    identity: 可选的已解码配置身份函数，重复配置不调用 fitness。
    restart_after: TE-RIME 连续完整停滞轮数，默认关闭；重启评估计入 budget。
    grid    : 各维候选值个数（S5 坐标精修用），None 则不做精修。
    refine  : 是否启用 S5 坐标精修（仅 terime 有效）。
    """
    rng = np.random.RandomState(seed)
    algo = algo.lower()
    budget = int(pop * iters if budget is None else budget)
    if pop < 1 or dim < 1 or iters < 1 or budget < pop:
        raise ValueError("Require dim/pop/iters >= 1 and budget >= pop")
    if algo not in ("rime", "terime", "pso", "bwo", "gwo"):
        raise ValueError(f"未知算法: {algo}")
    t0 = time.perf_counter()
    seen = set()
    proposals = 0
    duplicate_streak = 0
    restarts = 0
    qobl_restarts = 0
    probes = 0
    pool = _OpPool()
    fam_log = {"rime": [0, 0], "de": [0, 0], "elite": [0, 0]}   # [尝试, 成功]
    m_f, m_cr = 0.5, 0.5
    grid_arr = None if grid is None else [int(x) for x in grid]
    if grid_arr is not None and len(grid_arr) != dim:
        raise ValueError("grid 长度必须等于 dim")

    def evaluate_new(candidate, initializing=False, probe=False):
        nonlocal proposals, duplicate_streak
        proposals += 1
        if identity is None:
            return candidate, float(fitness(candidate))
        key = identity(candidate)
        if key in seen:
            if probe:
                # S5：探测命中已评估配置 → 直接放弃，不动 streak、不重采样
                return None
            duplicate_streak += 1
            if duplicate_streak < 20 and not initializing:
                return None
            for _ in range(10000):
                candidate = rng.uniform(0.0, 1.0, dim)
                proposals += 1
                key = identity(candidate)
                if key not in seen:
                    break
            else:
                raise RuntimeError("Unable to sample an unseen configuration")
        value = float(fitness(candidate))
        if not np.isfinite(value):
            raise ValueError("Fitness must be finite")
        seen.add(key)
        duplicate_streak = 0
        return candidate, value

    # 统一初始种群（所有算法共享，保证公平）；x0 提供时作为 warm-start
    # 注意：此处之前不得插入任何 rng 调用，否则五算法的初始种群会错位。
    if x0 is not None:
        X = np.asarray(x0, dtype=float)[:pop].copy()
        if X.ndim != 2 or X.shape[1] != dim or not np.isfinite(X).all():
            raise ValueError("x0 must have shape (k, dim) and finite values")
        if np.any((X < 0) | (X > 1)):
            raise ValueError("x0 must be within [0, 1]")
        if len(X) < pop:
            X = np.vstack([X, rng.uniform(0.0, 1.0, (pop - len(X), dim))])
    else:
        X = rng.uniform(0.0, 1.0, (pop, dim))
    initial_u = X.copy()
    fit = np.empty(pop)
    for i in range(pop):
        X[i], fit[i] = evaluate_new(X[i], initializing=True)
    best_i = int(np.argmin(fit))
    best, best_fit = X[best_i].copy(), float(fit[best_i])
    evals = pop
    hist = [dict(it=0, evals=evals, best_fit=best_fit, mean_fit=float(fit.mean()))]

    # PSO/BWO 状态
    vel = rng.uniform(-0.1, 0.1, (pop, dim))
    pbest, pbest_fit = X.copy(), fit.copy()

    stall = 0
    iteration = 0
    last_gen_evals = evals
    cdim, cdir = 0, +1

    # ------------------------------------------------------------------
    # S5：坐标精修（compass / Hooke-Jeeves 型）。只更新 best，不写种群。
    # ------------------------------------------------------------------
    def compass(max_probe):
        nonlocal best, best_fit, evals, probes, cdim, cdir
        used = 0
        while used < max_probe and evals < budget:
            j = cdim % dim
            n = grid_arr[j]
            improved = False
            for k in (cdir, -cdir):
                cand = _grid_step(best, j, n, k)
                if cand is None:
                    used += 1
                    continue
                got = evaluate_new(cand, probe=True)
                if got is None:                 # 该格点已评估过
                    used += 1
                    continue
                evals += 1
                used += 1
                probes += 1
                if got[1] < best_fit - 1e-12:
                    best, best_fit = got[0].copy(), float(got[1])
                    improved = True
                    cdir = k
                    # pattern move：沿成功方向再走一格
                    if evals < budget:
                        cand2 = _grid_step(best, j, n, k)
                        if cand2 is not None:
                            got2 = evaluate_new(cand2, probe=True)
                            if got2 is not None:
                                evals += 1
                                used += 1
                                probes += 1
                                if got2[1] < best_fit - 1e-12:
                                    best, best_fit = got2[0].copy(), float(got2[1])
                    break
            cdim += 0 if improved else 1
            if improved:
                # 该维仍在改善：下一代继续沿同维爬，加速逼近边界角点
                break
        return used

    while evals < budget:
        iteration += 1
        if identity is None and iteration > iters:
            break
        # S2：一律按评估进度推进，identity 模式下不会因迭代号封顶而冻结
        p = min(evals / float(budget), 0.999)
        prev_best = best_fit
        New = X.copy()
        fam_used = np.full(pop, -1)
        s_f, s_cr = [], []

        if algo in ("rime", "terime"):
            # RIME 原文：β = 2(1 − p) 不截断，但设地板避免后期零位移
            h = max(1.0 - p, 0.05)
            beta = max(2.0 * (1.0 - p), 0.15)
            E = p ** 0.75                      # 软/硬雾凇切换（随进度由硬转软）
            for i in range(pop):
                r1 = rng.uniform(-1, 1)
                theta = rng.uniform(0, 2 * np.pi)
                if rng.random() < E:           # 软雾凇：大范围游走
                    core = best + r1 * np.cos(theta) * beta * h
                else:                          # 硬雾凇：沿 best→个体方向穿刺
                    r2 = rng.uniform(-1, 1)
                    core = best + r2 * (best - X[i])
                if algo == "rime":
                    New[i] = _clip(core)
                    continue
                # ---- TE-RIME：S4 自适应算子池 ----
                k = pool.select(rng)
                fam_used[i] = k
                if k == 0:
                    v = core
                elif k == 1:
                    a, b, c = rng.choice(pop, 3, replace=False)
                    F = _sample_F(rng, m_f)
                    CR = _sample_CR(rng, m_cr)
                    if rng.random() < 0.5:
                        donor = X[a] + F * (X[b] - X[c])       # DE/rand/1
                    else:
                        donor = best + F * (X[a] - X[b])       # DE/best/1
                    v = _crossover_binomial(core, donor, rng, CR)
                else:
                    sigma = 0.15 * max(1.0 - p, 0.1)
                    if rng.random() < 0.5:
                        v = _crossover_binomial(core, best + sigma * rng.randn(dim), rng, 0.9)
                    else:
                        v = _crossover_binomial(core, _qobl(best, rng), rng, 0.9)
                New[i] = _reflect(v)           # S3

        elif algo == "pso":
            w, c1, c2 = 0.7298, 1.4962, 1.4962
            for i in range(pop):
                r1, r2 = rng.random(dim), rng.random(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (best - X[i])
                vel[i] = np.clip(vel[i], -0.3, 0.3)
                New[i] = _clip(X[i] + vel[i])

        elif algo == "bwo":
            # S7：Zhong et al. 2022 原式。Bf = B0(1 − p)，Bf>0.5 探索、否则捕食
            Wf = 0.1 - 0.05 * p
            Bf = rng.random() * (1.0 - p)
            for i in range(pop):
                if Bf > 0.5:                                   # 探索：同步/镜像游泳
                    r1, r2 = rng.random(), rng.random()
                    r = rng.randint(pop)
                    if rng.random() < 0.5:
                        v = X[i] + (X[r] - X[i]) * (1.0 + r1) * np.sin(2 * np.pi * r2)
                    else:
                        v = X[i] + (best - X[i]) * (1.0 + r1) * np.cos(2 * np.pi * r2)
                else:                                          # 开发：猎物围攻 + Lévy
                    r3, r4 = rng.random(), rng.random()
                    C1 = 2.0 * r4 * (1.0 - p)
                    LF = _levy(dim, rng)
                    r = rng.randint(pop)
                    v = r3 * best - r4 * X[i] + C1 * LF * (X[r] - X[i])
                if Bf * rng.random() < Wf:                     # 鲸落
                    C2 = 2.0 * Wf * pop
                    Xstep = np.exp(-C2 * p)
                    r5, r6, r7 = rng.random(), rng.random(), rng.random()
                    r = rng.randint(pop)
                    v = r5 * X[i] - r6 * X[r] + r7 * Xstep
                New[i] = _clip(v)

        elif algo == "gwo":
            # S7：补回 A = 2a·r1 − a 与 C = 2·r2
            a = 2.0 - 2.0 * p
            order = np.argsort(fit)
            Xa = X[order[0]]
            Xb = X[order[min(1, pop - 1)]]
            Xc = X[order[min(2, pop - 1)]]
            for i in range(pop):
                A1 = 2.0 * a * rng.random(dim) - a
                C1 = 2.0 * rng.random(dim)
                X1 = Xa - A1 * np.abs(C1 * Xa - X[i])
                A2 = 2.0 * a * rng.random(dim) - a
                C2 = 2.0 * rng.random(dim)
                X2 = Xb - A2 * np.abs(C2 * Xb - X[i])
                A3 = 2.0 * a * rng.random(dim) - a
                C3 = 2.0 * rng.random(dim)
                X3 = Xc - A3 * np.abs(C3 * Xc - X[i])
                New[i] = _clip((X1 + X2 + X3) / 3.0)
        else:
            raise ValueError(f"未知算法: {algo}")

        # 评估新种群（评估预算按唯一配置评估次数计）
        for i in range(pop):
            if evals >= budget:
                New[i] = X[i]
                fit[i] = fit[i]
                continue
            evaluated = evaluate_new(New[i])
            if evaluated is None:
                continue
            New[i], f_new = evaluated
            evals += 1
            improved = bool(f_new < fit[i])
            if improved:                                   # 正向贪婪选择
                X[i] = New[i]
                fit[i] = f_new
                if f_new < best_fit:
                    best, best_fit = New[i].copy(), float(f_new)
            if f_new < pbest_fit[i]:                       # PSO/BWO 个体历史最优
                pbest[i], pbest_fit[i] = New[i].copy(), f_new
            if fam_used[i] >= 0:                           # S4 算子收益统计
                name = _OpPool.NAMES[fam_used[i]]
                fam_log[name][0] += 1
                if improved:
                    fam_log[name][1] += 1
                pool.update(fam_used[i], improved)
                if improved and fam_used[i] == 1:
                    s_f.append(_sample_F(rng, m_f))
                    s_cr.append(_sample_CR(rng, m_cr))

        # S4：SHADE 记忆更新（胜出 F/CR 的 Lehmer 均值）
        if s_f:
            m_f = _lerp_mean(s_f)
            m_cr = float(np.clip(np.mean(s_cr), 0.0, 1.0))

        # S5：坐标精修（每代预留 1 槽；停滞≥2 代改为全维扫）
        if algo == "terime" and refine and grid_arr is not None and identity is not None:
            if evals < budget:
                compass(1 if stall < 2 else 2 * dim)

        # 显式代计数：compass 占用评估槽后 evals 增量不再恒等于 pop
        gen_evals = evals - last_gen_evals
        last_gen_evals = evals
        full_round = gen_evals >= pop
        if best_fit < prev_best - 1e-12:
            stall = 0
        elif full_round:
            stall += 1

        # S6：准对立重启（保留精英，替换最差 2 个，计入预算）
        if algo == "terime" and restart_after and stall >= restart_after:
            elite = int(np.argmin(fit))
            worst = [i for i in np.argsort(fit)[::-1] if i != elite][:2]
            for i in worst:
                if evals >= budget:
                    break
                got = evaluate_new(_qobl(best, rng), probe=True)
                if got is None:
                    continue
                X[i], fit[i] = got
                pbest[i], pbest_fit[i] = got[0].copy(), got[1]
                evals += 1
                qobl_restarts += 1
                if got[1] < best_fit:
                    best, best_fit = got[0].copy(), float(got[1])
            last_gen_evals = evals
            stall = 0

        hist.append(dict(it=iteration, evals=evals, best_fit=best_fit,
                         mean_fit=float(fit.mean())))
        if patience and stall >= patience:
            log(f"    [{algo}] 停在 it={iteration}（连续 {patience} 代无改进）")
            break

    return dict(algo=algo, best_u=best, best_fit=float(best_fit), history=hist,
                evals=int(evals), wall_sec=round(time.perf_counter() - t0, 2), seed=seed,
                pop=pop, iters=iters, budget=budget, proposals=proposals,
                restarts=restarts + qobl_restarts, qobl_restarts=qobl_restarts,
                probes=probes, op_stats=pool.as_dict(), op_family_log=fam_log,
                initial_u=initial_u.tolist(), algorithm_version=ALGORITHM_VERSION)


if __name__ == "__main__":
    # 自检：在 Rastrigin 上比较收敛，并核对确定性与预算精确性
    def fit(u):
        x = u * 10.24 - 5.12
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    grid = [8] * 6
    for a in ("rime", "terime", "pso", "bwo", "gwo"):
        r = run(a, 6, fit, pop=5, iters=60, seed=1, log=lambda *x, **k: None,
                grid=grid, restart_after=3)
        print(f"{a:8s} best={r['best_fit']:.4f} evals={r['evals']} {r['wall_sec']}s "
              f"probes={r['probes']} qobl={r['qobl_restarts']}")
    r1 = run("terime", 6, fit, pop=5, iters=40, seed=7, log=lambda *x, **k: None,
             grid=grid, restart_after=3)
    r2 = run("terime", 6, fit, pop=5, iters=40, seed=7, log=lambda *x, **k: None,
             grid=grid, restart_after=3)
    assert r1["history"] == r2["history"], "非确定性：两次运行轨迹不一致"
    assert r1["evals"] == r1["budget"], f"预算不精确：{r1['evals']} != {r1['budget']}"
    mono = all(b["best_fit"] <= a["best_fit"] + 1e-12
               for a, b in zip(r1["history"], r1["history"][1:]))
    assert mono, "best-so-far 非单调"
    print("自检通过：确定性 / 预算精确 / 单调")
