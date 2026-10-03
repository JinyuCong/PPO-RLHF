# =============================================================================
# plot_results.py —— 从 TensorBoard 日志画训练结果图
#
# 输出两张图：
#   1. training_dashboard.png —— 2×4 监控面板（奖励 / KL / 熵 / 长度 / Critic / PPO 更新健康度）
#   2. reward_vs_kl.png       —— Reward–KL 散点图，颜色表示训练步数
#
# 运行方式：
#   python plot_results.py                       # 默认画 runs/ 下最新的一次实验
#   python plot_results.py --run runs/20261003_113410
#   python plot_results.py --smooth 0.8 --out figures
# =============================================================================

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# 配色：一个主色画主线，文字/坐标轴用中性灰，不用彩色文字
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load_scalars(run_dir: str) -> dict:
    """读取一个 run 目录下所有 scalar，返回 {tag: (steps, values)}"""
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    data = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        data[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return data


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    """和 TensorBoard 一样的指数滑动平均（带 debias），weight=0 表示不平滑"""
    out, last = [], 0.0
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        out.append(last / (1 - weight ** (i + 1)))
    return np.array(out)


def pick(data: dict, *tags, negate_fallback: str = None):
    """按优先级取第一个存在的 tag；旧日志没有新指标时自动回退"""
    for t in tags:
        if t in data:
            return data[t], t
    if negate_fallback and negate_fallback in data:
        s, v = data[negate_fallback]
        return (s, -v), f"-{negate_fallback}"
    return None, None


def style_axes(ax, title: str, ylabel: str = None):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("PPO step", fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_curve(ax, series, smooth: float, band=None):
    """原始值画浅色细线，平滑值画 2px 主线；band 给出 ±std 阴影带"""
    steps, values = series
    if band is not None:
        _, std = band
        ax.fill_between(steps, values - std, values + std, color=SERIES, alpha=0.12, linewidth=0)
    ax.plot(steps, values, color=SERIES, alpha=0.3, linewidth=1)
    ax.plot(steps, ema(values, smooth), color=SERIES, linewidth=2)


def plot_missing(ax, title: str):
    style_axes(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "not logged in this run", transform=ax.transAxes,
            ha="center", va="center", color=TEXT_SECONDARY, fontsize=9)


def plot_dashboard(data: dict, smooth: float, out_path: str, run_name: str):
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    axes = axes.flatten()

    # (标题, y 轴含义, 候选 tag, 取负回退 tag, std 阴影带 tag)
    panels = [
        ("Reward-model score", "higher is better", ["reward/rm_score", "reward/mean"], None, "reward/rm_score_std"),
        ("KL to reference", "KL(π‖π_ref)", ["kl/mean"], None, None),
        ("Policy entropy", "nats / token", ["policy/entropy"], "loss/entropy", None),
        ("Response length", "tokens", ["response/length"], None, None),
        ("Critic loss", "MSE", ["loss/critic"], None, None),
        ("Critic explained variance", "1 = perfect", ["critic/explained_var"], None, None),
        ("PPO clip fraction", "fraction of tokens", ["policy/clip_frac"], None, None),
        ("Approx KL (new ‖ old)", "per update", ["policy/approx_kl"], None, None),
    ]

    for ax, (title, ylabel, tags, neg_tag, band_tag) in zip(axes, panels):
        series, used = pick(data, *tags, negate_fallback=neg_tag)
        if series is None:
            plot_missing(ax, title)
            continue
        band = data.get(band_tag) if band_tag and used == tags[0] else None
        plot_curve(ax, series, smooth, band)
        # 回退到旧 tag 时在标题里注明，避免误读
        style_axes(ax, title if used == tags[0] else f"{title}  [{used}]", ylabel)

    fig.suptitle(f"PPO-RLHF training · {run_name}", x=0.01, ha="left",
                 fontsize=13, color=TEXT_PRIMARY)
    fig.text(0.99, 0.965, f"thin line = raw, thick line = EMA({smooth})",
             ha="right", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_reward_vs_kl(data: dict, out_path: str, run_name: str):
    reward, used = pick(data, "reward/rm_score", "reward/mean")
    kl, _ = pick(data, "kl/mean")
    if reward is None or kl is None:
        print("skip reward_vs_kl: missing reward or kl")
        return

    # 按 step 对齐两条曲线
    steps = np.intersect1d(reward[0], kl[0])
    r = dict(zip(*reward))
    k = dict(zip(*kl))
    xs = np.array([k[s] for s in steps])
    ys = np.array([r[s] for s in steps])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(xs, ys, color=GRID, linewidth=1, zorder=1)
    # 截掉 Blues 最浅的一段，保证第一个点在白底上也看得见
    cmap = LinearSegmentedColormap.from_list("blues_trunc", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    sc = ax.scatter(xs, ys, c=steps, cmap=cmap,
                    s=60, edgecolors="white", linewidths=1.5, zorder=2)
    # 只标首尾两个点
    for i, label, dx, ha in ((0, f"step {steps[0]}", 8, "left"), (-1, f"step {steps[-1]}", -8, "right")):
        ax.annotate(label, (xs[i], ys[i]), xytext=(dx, 8), textcoords="offset points",
                    ha=ha, fontsize=8, color=TEXT_SECONDARY)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("PPO step", fontsize=9, color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    style_axes(ax, f"Reward vs. KL · {run_name}", "reward-model score" if used == "reward/rm_score" else used)
    ax.set_xlabel("KL to reference policy", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="TensorBoard run 目录，默认取 runs/ 下最新的")
    parser.add_argument("--out", default="figures", help="图片输出目录")
    parser.add_argument("--smooth", type=float, default=0.6, help="EMA 平滑系数，0~1")
    args = parser.parse_args()

    run_dir = args.run
    if run_dir is None:
        runs = sorted(d for d in glob.glob("runs/*") if os.path.isdir(d))
        if not runs:
            raise SystemExit("runs/ 下没有找到实验记录")
        run_dir = runs[-1]
    run_name = os.path.basename(os.path.normpath(run_dir))

    data = load_scalars(run_dir)
    if not data:
        raise SystemExit(f"{run_dir} 里没有 scalar 数据")
    print(f"loaded {run_dir}: {', '.join(sorted(data))}")

    os.makedirs(args.out, exist_ok=True)
    plot_dashboard(data, args.smooth, os.path.join(args.out, "training_dashboard.png"), run_name)
    plot_reward_vs_kl(data, os.path.join(args.out, "reward_vs_kl.png"), run_name)


if __name__ == "__main__":
    main()
