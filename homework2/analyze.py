#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业二 · 数据分析脚本
=====================================================================
输入：hw2_HD7CE2_*.csv（实验室平台下载的逐试次数据）
输出：results/ 目录下的统计表（csv / markdown）与两张图

分析口径（同时写进 README 的方法部分）：
  1. 同一被试编号若多次提交，只保留最早的一次（P05 有两次提交，剔除后一次）
  2. 反应时只统计「正确」试次；正确率统计全部试次
  3. 极端反应时剔除：< 150 ms 或 > 该任务上限（Stroop 3000 / 比例 5000）
  4. 组水平推断：以「被试」为单位做配对比较
       · 符号翻转置换检验（精确枚举 2^n 种符号组合）——不依赖正态假设
       · 符号检验（只看方向一致性）作为补充
       · 报告 Cohen's dz
  5. 由于本数据存在明显的速度-准确性权衡，额外报告
       IES（反效率分数）= 平均正确反应时 / 正确率，把速度与准确合并为一个指标
       （Townsend & Ashby, 1978）
=====================================================================
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "hw2_HD7CE2_2026-11-04_160646.csv"
OUT_DIR = BASE_DIR / "results"
if not OUT_DIR.exists():
    OUT_DIR.mkdir()

RT_MIN = 150
RT_MAX = {"stroop": 3000, "ratio": 5000}

C_CON, C_INC = "#f0a6a6", "#b03030"   # Stroop 一致 / 不一致
C_NT, C_WT = "#b8c6d9", "#2b6cb0"     # 比例 无刻度 / 有刻度


# ---------------------------------------------------------------------
# 1. 读入与清洗
# ---------------------------------------------------------------------
def load_data(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    log = {}

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    log["原始试次行数"] = len(df)
    log["原始被试编号数"] = df["participant"].nunique()

    # 同一编号多次提交 → 只保留最早一次
    first_session = (df.sort_values("created_at")
                       .groupby("participant", as_index=False)
                       .first()[["participant", "record_id"]])
    keep = set(zip(first_session["participant"], first_session["record_id"]))
    dup = ~df.apply(lambda r: (r["participant"], r["record_id"]) in keep, axis=1)
    log["重复提交的被试"] = sorted(df.loc[dup, "participant"].unique().tolist())
    log["剔除的重复试次"] = int(dup.sum())
    df = df.loc[~dup].copy()

    log["缺失反应时（超时）试次"] = int(df["rt"].isna().sum())

    def is_extreme(row):
        if pd.isna(row["rt"]):
            return True
        return row["rt"] < RT_MIN or row["rt"] > RT_MAX.get(row["exp"], 99999)

    ext = df.apply(is_extreme, axis=1)
    log["极端/缺失反应时试次"] = int(ext.sum())
    df["rt_clean"] = df["rt"].where(~ext)

    log["纳入分析被试数"] = df["participant"].nunique()
    log["纳入分析试次数"] = int(len(df))
    return df, log


# ---------------------------------------------------------------------
# 2. 统计工具
# ---------------------------------------------------------------------
def signflip_test(diffs):
    """符号翻转置换检验（双侧，精确枚举）→ p"""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    obs = abs(d.mean())
    hits = sum(abs((d * np.array(s)).mean()) >= obs - 1e-12
               for s in itertools.product([1, -1], repeat=n))
    return hits / (2 ** n)


def sign_test(diffs):
    """符号检验（双侧，二项精确）"""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d) & (d != 0)]
    n = len(d)
    k = min((d > 0).sum(), (d < 0).sum())
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def effect_stats(diffs):
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    dz = d.mean() / sd if sd > 0 else np.nan
    return t, dz


def p_text(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def subject_table(df, exp, conds):
    """每名被试 × 每条件：平均正确反应时、正确率、IES"""
    sub = df[df["exp"] == exp]
    rows = []
    for pid, g in sub.groupby("participant"):
        rec = {"participant": pid}
        for c in conds:
            gc = g[g["cond"] == c]
            rt = gc["rt_clean"].dropna()
            rec[f"rt_{c}"] = rt.mean() if len(rt) else np.nan
            rec[f"acc_{c}"] = gc["correct"].mean() if len(gc) else np.nan
            # IES：把速度与准确合并（越大越差）
            rec[f"ies_{c}"] = (rec[f"rt_{c}"] / rec[f"acc_{c}"]
                               if rec[f"acc_{c}"] and not np.isnan(rec[f"rt_{c}"]) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)


def compare(name, sub, c1, c2, metric, expected_sign):
    """
    对比两个条件。
    expected_sign: 预测的方向（差 = 条件2 − 条件1 的符号），用于统计"方向一致人数"
    """
    a = sub[f"{metric}_{c1}"].values.astype(float)
    b = sub[f"{metric}_{c2}"].values.astype(float)
    if metric == "acc":          # 正确率统一换算成百分数，便于阅读
        a, b = a * 100, b * 100
    d = b - a
    t, dz = effect_stats(d)
    valid = d[~np.isnan(d)]
    consistent = int((np.sign(valid) == expected_sign).sum())
    return dict(
        比较=name,
        条件一=np.nanmean(a),
        条件二=np.nanmean(b),
        差值=np.nanmean(d),
        t=t, dz=dz,
        p_perm=signflip_test(d),
        p_sign=sign_test(d),
        方向一致=f"{consistent}/{len(valid)}",
        预期方向="＋" if expected_sign > 0 else "－",
    )


# ---------------------------------------------------------------------
# 3. 作图
# ---------------------------------------------------------------------
def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", color="#e8ecf2", linewidth=0.8)
    ax.set_axisbelow(True)


def paired_panel(ax, v1, v2, labels, colors, ylabel, title, ylim=None):
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    means = [np.nanmean(v1), np.nanmean(v2)]
    sems = [np.nanstd(v1, ddof=1) / np.sqrt(np.sum(~np.isnan(v1))),
            np.nanstd(v2, ddof=1) / np.sqrt(np.sum(~np.isnan(v2)))]

    ax.bar([0, 1], means, width=0.5, color=colors, edgecolor="none", alpha=0.85, zorder=1)
    ax.errorbar([0, 1], means, yerr=sems, fmt="none", ecolor="#333",
                capsize=6, linewidth=1.4, zorder=3)

    for a, b in zip(v1, v2):
        if np.isnan(a) or np.isnan(b):
            continue
        ax.plot([0, 1], [a, b], color="#9aa4b2", linewidth=0.85, alpha=0.6, zorder=2)
        ax.scatter([0, 1], [a, b], s=20, color="#4a5260", zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=12.5, pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    style_axis(ax, ylabel)


def make_figures(stroop_sub, ratio_sub):
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    # 图 1：Stroop —— 原始反应时 & IES
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200)
    paired_panel(axes[0], stroop_sub["rt_congruent"], stroop_sub["rt_incongruent"],
                 ["Congruent", "Incongruent"], [C_CON, C_INC],
                 "Reaction time (ms)", "Stroop: raw RT")
    paired_panel(axes[1], stroop_sub["ies_congruent"], stroop_sub["ies_incongruent"],
                 ["Congruent", "Incongruent"], [C_CON, C_INC],
                 "IES (ms)", "Stroop: inverse efficiency score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_stroop.png", bbox_inches="tight")
    plt.close(fig)

    # 图 2：比例判断 —— 反应时 & 正确率
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200)
    paired_panel(axes[0], ratio_sub["rt_no_ticks"], ratio_sub["rt_with_ticks"],
                 ["No ticks", "With ticks"], [C_NT, C_WT],
                 "Reaction time (ms)", "Ratio judgment: RT")
    paired_panel(axes[1], ratio_sub["acc_no_ticks"] * 100, ratio_sub["acc_with_ticks"] * 100,
                 ["No ticks", "With ticks"], [C_NT, C_WT],
                 "Accuracy (%)", "Ratio judgment: accuracy", ylim=(60, 106))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_ratio.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------
def main():
    df, log = load_data(CSV_PATH)

    print("=" * 72)
    print("① 数据清洗")
    print("=" * 72)
    for k, v in log.items():
        print(f"  {k}: {v}")

    stroop_sub = subject_table(df, "stroop", ["congruent", "incongruent"])
    ratio_sub = subject_table(df, "ratio", ["no_ticks", "with_ticks"])
    stroop_sub.to_csv(OUT_DIR / "per_subject_stroop.csv", index=False, encoding="utf-8-sig")
    ratio_sub.to_csv(OUT_DIR / "per_subject_ratio.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("② 描述统计（每名被试，反应时仅正确试次）")
    print("=" * 72)
    print(stroop_sub.round(1).to_string(index=False))
    print()
    print(ratio_sub.round(1).to_string(index=False))

    # ---------- 推断统计 ----------
    rows = [
        compare("Stroop 反应时：不一致 − 一致", stroop_sub, "congruent", "incongruent", "rt", +1),
        compare("Stroop 正确率：不一致 − 一致", stroop_sub, "congruent", "incongruent", "acc", -1),
        compare("Stroop IES：不一致 − 一致", stroop_sub, "congruent", "incongruent", "ies", +1),
        compare("比例 反应时：无刻度 − 有刻度", ratio_sub, "no_ticks", "with_ticks", "rt", +1),
        compare("比例 正确率：有刻度 − 无刻度", ratio_sub, "with_ticks", "no_ticks", "acc", +1),
    ]
    res = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("③ 配对检验（n = %d）" % len(stroop_sub))
    print("=" * 72)
    for r in rows:
        unit = "%" if "正确率" in r["比较"] else ""
        print(f"{r['比较']}")
        print(f"   {r['条件一']:.3f}{unit}  →  {r['条件二']:.3f}{unit}   差 = {r['差值']:+.2f}{unit}"
              f"   t = {r['t']:+.2f}   置换 p {p_text(r['p_perm'])}   符号 p {p_text(r['p_sign'])}"
              f"   dz = {r['dz']:+.2f}   方向一致 {r['方向一致']}")

    # ---------- markdown 表 ----------
    md = ["| 比较 | 条件一 | 条件二 | 差值 | t | 置换检验 p | 符号检验 p | Cohen's dz | 方向一致 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        unit = "%" if "正确率" in r["比较"] else ""
        md.append(f"| {r['比较']} | {r['条件一']:.1f}{unit} | {r['条件二']:.1f}{unit} | "
                  f"{r['差值']:+.1f}{unit} | {r['t']:+.2f} | {p_text(r['p_perm'])} | "
                  f"{p_text(r['p_sign'])} | {r['dz']:+.2f} | {r['方向一致']} |")
    (OUT_DIR / "summary_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))

    make_figures(stroop_sub, ratio_sub)
    print("\n图已保存到:", OUT_DIR)

    # ---------- 个体明细 ----------
    detail = stroop_sub[["participant", "rt_congruent", "rt_incongruent",
                         "acc_congruent", "acc_incongruent",
                         "ies_congruent", "ies_incongruent"]].copy()
    detail["stroop_RT干扰量"] = detail["rt_incongruent"] - detail["rt_congruent"]
    detail["stroop_IES差"] = detail["ies_incongruent"] - detail["ies_congruent"]
    detail = detail.merge(
        ratio_sub[["participant", "rt_no_ticks", "rt_with_ticks",
                   "acc_no_ticks", "acc_with_ticks"]], on="participant", how="outer")
    detail["刻度改善量"] = detail["rt_no_ticks"] - detail["rt_with_ticks"]
    detail.round(1).to_csv(OUT_DIR / "per_subject_effects.csv", index=False, encoding="utf-8-sig")
    print("\n④ 个体效应明细：")
    print(detail.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
