"""Draw the README chart of the Run 3 replay's cash against the live game (five seeds).

Usage: uv run --with matplotlib python verification/replay_cash_chart.py [out.png]
"""

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from scgame.engine import Engine  # noqa: E402
from scgame.history import read_history_xlsx, replay  # noqa: E402
from scgame.scenario import Scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots" / "replay-cash.png"
DATA = ROOT / "reference" / "data"
INK, ACCENT, MUTED, HAIR, BAND = "#1e1b2e", "#4338ca", "#6b6880", "#e6e4ec", "#f4f3f8"

scenario = Scenario.load(ROOT / "scenarios" / "classic-network.json")
decisions = read_history_xlsx(DATA / "run3_history.xlsx", {r.name: r.id for r in scenario.regions})
live = {int(r["day"]): float(r["cash_thousands"]) * 1000 for r in csv.DictReader(open(DATA / "run3_live_cash.csv"))}

runs = []
for seed in range(5):
    engine = Engine(scenario, seed=seed)
    assert replay(engine, decisions) == []
    runs.append({r.day: r.cash for r in engine.records})

days = [d for d in range(731, 1461) if d in live]
sim = [statistics.mean(run[d] for run in runs) for d in days]
real = [live[d] for d in days]
gap = [s - r for s, r in zip(sim, real)]
gaps = [[run[d] - live[d] for d in days] for run in runs]
print(f"final replay {sim[-1]:,.0f} live {real[-1]:,.0f} ({sim[-1] / real[-1] - 1:+.2%})")
print("largest gap ($k):", max(abs(g) for row in gaps for g in row) / 1e3)

for path in ("/System/Library/Fonts/SFNS.ttf",):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
plt.rcParams.update({"font.size": 13, "axes.edgecolor": HAIR, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0})

fig, (top, low) = plt.subplots(2, 1, figsize=(11.52, 5.6), dpi=200, sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.2})
fig.patch.set_facecolor("white")
for ax in (top, low):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.grid(axis="y", color=HAIR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvspan(730.5, 800.5, color=BAND, zorder=0, linewidth=0)

top.plot(days, real, color=INK, linewidth=1.6, label="Live game")
top.plot(days, sim, color=ACCENT, linewidth=1.6, label="Replay, five seeds")
top.set_ylim(0, 30e6)
top.set_yticks([0, 10e6, 20e6, 30e6])
top.set_yticklabels(["0", "10", "20", "30"])
top.set_ylabel("Cash ($M)")
top.text(765.5, 28.6e6, "First ten\nweeks", ha="center", va="top", color=MUTED, fontsize=11, linespacing=1.25)
top.text(1466, sim[-1], f"${sim[-1] / 1e6:.2f}M", color=INK, va="bottom", fontsize=12.5)
top.text(1466, real[-1], f"${real[-1] / 1e6:.2f}M", color=INK, va="top", fontsize=12.5)
legend = top.legend(loc="upper left", bbox_to_anchor=(0.125, 1.0), frameon=False, ncol=2, fontsize=12.5,
                    handlelength=1.6, columnspacing=2.0)
for t in legend.get_texts():
    t.set_color(INK)

low.axhline(0, color=INK, linewidth=0.8)
for row in gaps:
    low.plot(days, [g / 1e6 for g in row], color=ACCENT, linewidth=0.9, alpha=0.55)
low.set_ylabel("Gap ($M)")
low.set_ylim(-1.2, 2.6)
low.set_yticks([-1, 0, 1, 2])
low.set_yticklabels(["\u22121", "0", "1", "2"])
low.set_xlim(730, 1460)
low.set_xticks([730, 900, 1100, 1300, 1460])
low.set_xlabel("Day")

fig.subplots_adjust(left=0.07, right=0.915, top=0.97, bottom=0.12)
fig.savefig(OUT, facecolor="white")
print("saved", OUT)
