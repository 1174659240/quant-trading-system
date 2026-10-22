"""阶段 4.5：软权重持仓 + 成本感知再平衡（方法层改造，规则未调参）。

两处改动（预注册，动手前写死）：

1. 软权重持仓：调仓日分别构造 top40 / top50 / top60 三套等权目标，取三者平均。
   排名靠前者在三套里都出现→权重 3/3；边界处只出现在 top60→权重 1/3，自然平滑递减。
   其余规则（行业门类上限 30%、月频、gross 95%、<5000 元过滤、分档滑点、E 层）不变。

2. 成本感知再平衡：逐日比较「当前权重 → 目标权重」的偏离与交易成本。
   每只股票的偏离 d_i，年化跟踪误差 TE = sqrt(Σ d_i²σ_i²)（对角近似），
   修正该偏离带来的 TE 下降 ≈ d_i²σ_i²/(2·TE)；按策略自身的再平衡周期 T_r=1/12 年折算，
   当 T_r·d_i²σ_i²/(2TE) > c_i·|d_i| 才交易，即 |d_i| > 2·c_i·TE/(T_r·σ_i²)。
   c_i 由既有成本参数直接给出（佣金 2.5bp + 过户 0.1bp + 卖出印花税 5bp + 对应流动性档位滑点），
   没有任何需要标定的自由参数。σ_i 用 250 日已实现波动。

用法：
    python -m scripts.growth_alpha --runs v2a,v2b           # 只跑两个新版本
    python -m scripts.growth_alpha --runs v2a,v2b --robust  # 再跑阶段 4 七项检验（v2）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant.strategy.base import Strategy

OUT = Path("results/growth/v2")
ROBUST = Path("results/growth/robustness")
ROBUST_V2 = Path("results/growth/robustness-v2")

# ---- 预注册常量（不看结果再改）---------------------------------------------
KS = (40, 50, 60)              # 三套持仓数取平均
HORIZON_YEARS = 1.0 / 12.0     # 策略自身再平衡周期：月频
VOL_WINDOW = 250
VOL_MIN_OBS = 60
COMMISSION = 0.00025           # 与 quant/backtest/costs.py 一致
TRANSFER = 1e-5
STAMP = 0.0005                 # 仅卖出
DEFAULT_SLIP_BP = 3.44         # 分档滑点的全样本加权均值（阶段 3.1 实测），缺档时兜底
MIN_TRADE = 5000.0

RUN_SPECS: dict[str, dict] = {
    "v2a": {"ks": KS, "cost_aware": False, "label": "软权重（top40/50/60 平均）"},
    "v2b": {"ks": KS, "cost_aware": True, "label": "软权重 + 成本感知再平衡"},
}


# --------------------------------------------------------------------------- #
# 策略 1：软权重
# --------------------------------------------------------------------------- #
class SoftTopK(Strategy):
    """top40/50/60 三套等权目标的平均：边界平滑、不给参数留调节余地。"""

    name = "growth-soft-topk"

    def __init__(self, score: pd.DataFrame, eligible: pd.DataFrame, *, sector=None,
                 ks: tuple[int, ...] = KS, sector_cap: float = 0.30, gross: float = 0.95,
                 max_weight: float = 0.04, rebalance: str = "ME",
                 min_holdings: int = 10) -> None:
        super().__init__(rebalance=rebalance, gross=gross)
        from scripts.growth_backtest import GrowthTopK
        self.ks = tuple(int(k) for k in ks)
        self.parts = [GrowthTopK(score, eligible, sector=sector, top_k=k, gross=gross,
                                 max_weight=max_weight, sector_cap=sector_cap,
                                 rebalance=rebalance, min_holdings=min_holdings)
                      for k in self.ks]

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        total = None
        for part in self.parts:
            weights = part.generate(bars)
            total = weights if total is None else total.add(weights, fill_value=0.0)
        blended = total / float(len(self.parts))
        # 凸组合：每个子组合单票 ≤max_weight、行业 ≤cap、gross ≤0.95，故混合后同样满足
        return blended.clip(lower=0.0, upper=0.95)


# --------------------------------------------------------------------------- #
# 策略 2：成本感知再平衡
# --------------------------------------------------------------------------- #
class CostAwareStrategy(Strategy):
    """在目标权重与当前权重之间做「偏离 vs 成本」的逐日取舍。

    只做一件事：当把权重拉向目标所买到的年化跟踪误差下降（按月频周期折算）
    覆盖不了这一笔的往返成本时，就不交易，保持不动。
    """

    name = "growth-cost-aware"

    def __init__(self, base: Strategy, panel, *, marks, eligible, slip_maps: dict,
                 horizon_years: float = HORIZON_YEARS, vol_window: int = VOL_WINDOW) -> None:
        super().__init__(rebalance="ME", gross=0.95)
        self.base = base
        self.panel = panel
        self.marks = pd.DatetimeIndex(marks)
        self.eligible = eligible
        self.slip_maps = slip_maps
        self.horizon = float(horizon_years)
        self.vol_window = int(vol_window)
        self.diagnostics: dict = {"days": 0, "trades_skipped": 0, "trades_taken": 0,
                                  "band_median": float("nan")}

    # -- 工具 ---------------------------------------------------------------
    def _sigma(self) -> pd.DataFrame:
        close = self.panel.close
        returns = close.pct_change()
        sigma = returns.rolling(self.vol_window, min_periods=VOL_MIN_OBS).std() * math.sqrt(244.0)
        return sigma

    def _slip_for(self, day: pd.Timestamp, symbol: str) -> float:
        # 取当日之前最近的月末档位（无则用全局均值）
        candidates = [d for d in self.marks if d <= day]
        if not candidates:
            return DEFAULT_SLIP_BP
        table = self.slip_maps.get(candidates[-1])
        if not table:
            return DEFAULT_SLIP_BP
        return float(table.get(symbol, DEFAULT_SLIP_BP))

    def _cost(self, day: pd.Timestamp, symbol: str, delta: float) -> float:
        slip = self._slip_for(day, symbol) / 1e4
        cost = COMMISSION + TRANSFER + slip
        if delta < 0:
            cost += STAMP
        return cost

    # -- 主逻辑 -------------------------------------------------------------
    def _slip_matrix(self, dates: pd.DatetimeIndex, cols: list[str]) -> np.ndarray:
        """每天每个名字的分档滑点（bp）：取当日之前最近的月末档位，缺失用全局均值。"""
        mark_positions = np.searchsorted(self.marks.values, dates.values, side="right") - 1
        cache: dict[int, np.ndarray] = {}
        out = np.full((len(dates), len(cols)), DEFAULT_SLIP_BP, dtype="float64")
        for idx, pos in enumerate(mark_positions):
            if pos < 0:
                continue
            if pos not in cache:
                table = self.slip_maps.get(self.marks[int(pos)], {}) or {}
                # 先把当档位所有名字的滑点铺满，再覆盖有档位的
                cache[pos] = np.array([float(table.get(s, DEFAULT_SLIP_BP)) for s in cols],
                                      dtype="float64")
            out[idx] = cache[pos]
        return out

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        target = self.base.generate(bars)
        dates = pd.DatetimeIndex(self.panel.dates)
        cols = list(target.columns)
        target = target.reindex(index=dates, columns=cols).ffill().fillna(0.0)
        sigma = self._sigma().reindex(index=dates, columns=cols)
        sigma = sigma.to_numpy(dtype="float64")
        row_med = np.nanmedian(sigma, axis=1)
        bad = ~np.isfinite(sigma) | (sigma <= 1e-8)
        if bad.any():
            fill = np.where(np.isfinite(row_med), row_med, 0.35)[:, None]
            sigma = np.where(bad, fill, sigma)
        slip = self._slip_matrix(dates, cols) / 1e4
        base_cost = COMMISSION + TRANSFER
        goal_all = target.to_numpy(dtype="float64")

        current = np.zeros(len(cols), dtype="float64")
        rows = np.zeros((len(dates), len(cols)), dtype="float64")
        bands: list[float] = []
        skipped = taken = 0
        for i in range(len(dates)):
            goal = goal_all[i]
            delta = goal - current
            vol = sigma[i]
            te = float(np.sqrt(np.nansum((delta * vol) ** 2)))
            cost = base_cost + slip[i] + np.where(delta < 0, STAMP, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                band = 2.0 * cost * te / (self.horizon * vol ** 2)
            trade = (np.abs(delta) > band) & (np.abs(delta) > 1e-9)
            new = np.where(trade, goal, current)
            skipped += int((~trade).sum())
            taken += int(trade.sum())
            rows[i] = new
            current = new
            if te > 0:
                bands.append(float(np.nanmedian(band)))
        self.diagnostics.update({"days": len(dates), "trades_skipped": skipped,
                                 "trades_taken": taken,
                                 "band_median": float(np.nanmedian(bands)) if bands else float("nan")})
        frame = pd.DataFrame(rows, index=dates, columns=cols)
        return frame.clip(lower=0.0, upper=0.95)


# --------------------------------------------------------------------------- #
# 构造器（供 growth_robust 的敏感性/WFA 复用）
# --------------------------------------------------------------------------- #
def make_builder(spec: dict, prepared: dict):
    from scripts import growth_risk as gr
    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    slip_maps, _ = gr.tier_map(prepared["marks"], prepared["eligible"], adv20, illiq20)

    def build(prep: dict, item: dict) -> pd.DataFrame:
        score = item.get("score")
        if score is None:
            score = prep["score"]
        soft = SoftTopK(score, prep["eligible"], sector=prep["sector"],
                        ks=item.get("ks", spec["ks"]),
                        sector_cap=item.get("sector_cap", prep["sector_cap"]))
        if not spec["cost_aware"]:
            return soft.generate(prep["bars"])
        wrapper = CostAwareStrategy(soft, prep["panel"], marks=prep["marks"],
                                    eligible=prep["eligible"], slip_maps=slip_maps)
        weights = wrapper.generate(prep["bars"])
        prep["cost_diag"] = dict(wrapper.diagnostics)
        return weights

    return build


def _slip_union(marks, eligible, adv20, illiq20) -> dict[str, float]:
    from scripts import growth_risk as gr
    slip_maps, _ = gr.tier_map(marks, eligible, adv20, illiq20)
    union: dict[str, float] = {}
    for day in marks:
        union.update(slip_maps.get(day, {}))
    return union


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(runs: list[str], *, robust: bool = False) -> dict:
    from scripts import growth_robust as grb
    from scripts import growth_risk as gr

    OUT.mkdir(parents=True, exist_ok=True)
    prepared = grb.prepare("main")
    marks = prepared["marks"]
    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    slip_maps, _ = gr.tier_map(marks, prepared["eligible"], adv20, illiq20)
    tier_slip = _slip_union(marks, prepared["eligible"], adv20, illiq20)

    rows = []
    for name in runs:
        spec = RUN_SPECS[name]
        started = time.time()
        build = make_builder(spec, prepared)
        weights = build(prepared, {})
        diagnostics = prepared.pop("cost_diag", None)
        outcome = grb.run_engine(prepared, weights, name=name, min_trade=MIN_TRADE,
                                 slip_map=tier_slip)
        row = outcome["row"]
        row["spec"] = name
        row["label"] = spec["label"]
        row["seconds"] = round(time.time() - started, 1)
        if diagnostics:
            row["cost_aware_skipped"] = diagnostics.get("trades_skipped")
            row["cost_aware_taken"] = diagnostics.get("trades_taken")
            row["cost_band_median"] = diagnostics.get("band_median")
        rows.append(row)
        daily = outcome["daily"].copy()
        daily.index.name = "date"
        daily.to_csv(OUT / f"daily-{name}.csv", encoding="utf-8-sig")
        weights.to_csv(OUT / f"weights-{name}.csv.gz", compression="gzip")
        print(f"[{name}] cagr {row['cagr']:.2%} sharpe {row['sharpe']:.3f} "
              f"mdd {row['max_drawdown']:.2%} turnover {row['ann_turnover_one_way']:.1%} "
              f"fees {row['fees_pct_of_equity']:.2%}/yr ({row['seconds']}s)", flush=True)
        if diagnostics:
            print(f"[{name}] 成本感知：跳过 {diagnostics['trades_skipped']:,} 个(日,票)，"
                  f"成交 {diagnostics['trades_taken']:,} 个，band 中位 {diagnostics['band_median']:.4f}",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    print(frame[["spec", "cagr", "sharpe", "max_drawdown", "ann_turnover_one_way",
                 "fees_pct_of_equity"]].to_string(index=False), flush=True)

    result = {"metrics": frame.to_dict(orient="records")}
    if robust:
        result["robustness"] = run_robustness_v2(runs, prepared=prepared)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                 default=str), encoding="utf-8")
    return result


def run_history(runs: list[str]) -> dict:
    """历史样本（2008-08~2016-12）补充：只做 v2b 的回测 + 统计 + bootstrap。"""
    from scripts import growth_robust as grb
    from scripts import growth_risk as gr

    out = OUT / "history"
    out.mkdir(parents=True, exist_ok=True)
    prepared = grb.prepare("history")
    marks = prepared["marks"]
    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    _, _ = gr.tier_map(marks, prepared["eligible"], adv20, illiq20)
    rows = []
    for name in runs:
        spec = RUN_SPECS[name]
        build = make_builder(spec, prepared)
        weights = build(prepared, {})
        outcome = grb.run_engine(prepared, weights, name=name, min_trade=MIN_TRADE)
        row = outcome["row"]
        row["spec"] = name
        row["label"] = spec["label"]
        rows.append(row)
        daily = outcome["daily"].copy()
        daily.index.name = "date"
        daily.to_csv(out / f"daily-{name}.csv", encoding="utf-8-sig")
        weights.to_csv(out / f"weights-{name}.csv.gz", compression="gzip")
        print(f"[history {name}] cagr {row['cagr']:.2%} sharpe {row['sharpe']:.3f} "
              f"mdd {row['max_drawdown']:.2%} turnover {row['ann_turnover_one_way']:.1%}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    equity = grb.load_daily(out / f"daily-{runs[-1]}.csv")
    returns = grb.to_returns(equity)
    monthly = grb.monthly_returns(equity)
    payload = {
        "stats": grb.stats_block(returns),
        "bootstrap": grb.stationary_bootstrap(monthly),   # 与阶段 4 同口径：月度、block=6 月
        "is_oos": grb.is_oos(returns, split="2012-12-31"),
    }
    (out / "tests-history-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def write_ledger_v2(runs: list[str], *, new_trials: int = 2) -> dict:
    """把 v2 的新增试验补登进台账，并用新的 N 重算 DSR。"""
    from scripts import growth_robust as grb

    base_path = ROBUST / "trial-ledger.json"
    ledger = json.loads(base_path.read_text(encoding="utf-8"))
    n_cfg = ledger.get("N", {})
    old = {k: int(n_cfg.get(k, 0)) for k in ("N_spec_only", "N_strategy_level", "N_all")}
    new = {k: v + new_trials for k, v in old.items()}
    daily = OUT / f"daily-{runs[-1]}.csv"
    equity = grb.load_daily(daily)
    returns = grb.to_returns(equity)          # 日频，与阶段 4 的 DSR 口径一致
    dsr = {}
    for tag, n in (("N_spec_only", new["N_spec_only"]),
                   ("N_strategy_level", new["N_strategy_level"]),
                   ("N_all", new["N_all"])):
        dsr[tag] = grb.deflated_sharpe(returns, n)
    payload = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "objective": "v2：软权重(top40/50/60 平均) + 成本感知再平衡 + 逐日交易 + 分档滑点 + E 默认层",
        "prior_N": old,
        "new_trials": new_trials,
        "new_trials_detail": [f"{name}（{RUN_SPECS[name]['label']}）" for name in runs],
        "N": new,
        "dsr": dsr,
        "policy": "沿用阶段 4：敏感性邻域点与 WFA 折单独登记，不计入选优 N；N 只增不减。",
        "sensitivity_registry": [
            "ks=(30,40,50)", "ks=(50,60,70)", "sector_cap=0.25", "sector_cap=1.0",
            "min_trade=0", "min_trade=10000", "slip=fixed2bp",
        ],
        "wfa_registry": ["2020", "2021", "2022", "2023", "2024"],
    }
    ROBUST_V2.mkdir(parents=True, exist_ok=True)
    (ROBUST_V2 / "trial-ledger-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def run_robustness_v2(runs: list[str], *, prepared: dict | None = None) -> dict:
    """把阶段 4 的七项检验在同一套口径下重跑到 v2 版本。"""
    from scripts import growth_robust as grb

    ROBUST_V2.mkdir(parents=True, exist_ok=True)
    grb.set_root(ROBUST_V2)
    daily_v2 = OUT / f"daily-{runs[-1]}.csv"   # 以最后一个（v2b）为检验对象
    fast = grb.run_fast(daily_path=daily_v2, tag="-v2")
    if prepared is None:
        prepared = grb.prepare("main")
    spec = RUN_SPECS[runs[-1]]
    grb.set_builder(make_builder(spec, prepared))
    heavy: dict = {}
    started = time.time()
    heavy["perm"] = grb.permutation_test(prepared["score"], prepared["panel"].close,
                                        prepared["marks"], prepared["eligible"], reps=1000)
    print(f"[perm] p={heavy['perm'].get('p_value')} ({time.time() - started:.0f}s)", flush=True)
    started = time.time()
    runs_sens = [
        {"ks": (30, 40, 50), "sector_cap": prepared["sector_cap"], "min_trade": 5000.0, "slip": "tiered"},
        {"ks": (50, 60, 70), "sector_cap": prepared["sector_cap"], "min_trade": 5000.0, "slip": "tiered"},
        {"ks": (40, 50, 60), "sector_cap": 0.25, "min_trade": 5000.0, "slip": "tiered"},
        {"ks": (40, 50, 60), "sector_cap": 1.0, "min_trade": 5000.0, "slip": "tiered"},
        {"ks": (40, 50, 60), "sector_cap": prepared["sector_cap"], "min_trade": 0.0, "slip": "tiered"},
        {"ks": (40, 50, 60), "sector_cap": prepared["sector_cap"], "min_trade": 10000.0, "slip": "tiered"},
        {"ks": (40, 50, 60), "sector_cap": prepared["sector_cap"], "min_trade": 5000.0, "slip": "fixed"},
    ]
    heavy["sensitivity"] = grb.sensitivity_scan(prepared, tag="v2", runs=runs_sens).to_dict(orient="records")
    print(f"[sens] {len(heavy['sensitivity'])} 行 ({time.time() - started:.0f}s)", flush=True)
    started = time.time()
    heavy["wfa"] = grb.wfa(prepared, tag="v2")
    print(f"[wfa] ({time.time() - started:.0f}s)", flush=True)
    heavy["ledger"] = write_ledger_v2(runs, new_trials=len(runs) + len(runs_sens) + len(heavy["wfa"].get("folds", [])))
    (ROBUST_V2 / "tests-heavy-v2.json").write_text(
        json.dumps(heavy, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"fast": fast, "heavy": heavy}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="v2a,v2b")
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--robust-only", action="store_true",
                        help="跳过组合回测，只补跑重检验（复用已落盘的 daily-v2b.csv）")
    parser.add_argument("--sample", default="main", choices=["main", "history"],
                        help="history：历史样本补充（只跑回测+统计+bootstrap）")
    args = parser.parse_args(argv)
    runs = [part.strip() for part in args.runs.split(",") if part.strip()]
    for name in runs:
        if name not in RUN_SPECS:
            raise SystemExit(f"未知规格 {name}，可选 {sorted(RUN_SPECS)}")
    if args.sample == "history":
        payload = run_history(runs)
        print(json.dumps({k: (v if not isinstance(v, dict) else list(v)[:6]) for k, v in payload.items()},
                         ensure_ascii=False, default=str), flush=True)
        return 0
    if args.robust_only:
        from scripts import growth_robust as grb
        prepared = grb.prepare("main")
        payload = run_robustness_v2(runs, prepared=prepared)
        print(json.dumps({"robustness_keys": list(payload)}, ensure_ascii=False), flush=True)
        return 0
    run(runs, robust=args.robust)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
