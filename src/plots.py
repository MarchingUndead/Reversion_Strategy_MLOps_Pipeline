import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------- plot ----------
# Left y-axis: spread (%). Right y-axis: cash LTP. Stitched x-axis (holidays +
# overnight gaps removed). Multi-day view labels each session start; single-day
# view labels hourly boundaries and puts the date in the title.

def plot(df, events, symbol, contract, out_thresh, rev_thresh):
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(df))
    ax.plot(x, df["spread"].to_numpy(),
            color="steelblue", linewidth=0.5, zorder=3)

    # Cash LTP on a twin y-axis.
    ax2 = ax.twinx()
    ax2.plot(x, df["ltp"].to_numpy(),
             color="darkorange", linewidth=0.5, alpha=0.6, zorder=2)
    ax2.set_ylabel("cash LTP", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Bucket-piecewise threshold bands (on the spread axis).
    key    = df["dte"].astype(str) + "_" + df["bucket"].astype(str)
    seg_id = key.ne(key.shift()).cumsum()
    for _, seg in df.groupby(seg_id):
        m, s = seg["dist_mean"].iloc[0], seg["dist_std"].iloc[0]
        if pd.isna(m) or pd.isna(s):
            continue
        x0, x1 = seg.index[0], seg.index[-1]
        for k, color, style in [
            (out_thresh,              "black", "-"),
            (out_thresh + rev_thresh, "red",   "--"),
            (out_thresh - rev_thresh, "green", "--"),
        ]:
            ax.hlines(m + k*s, x0, x1, colors=color, linestyles=style, linewidth=0.7)
            ax.hlines(m - k*s, x0, x1, colors=color, linestyles=style, linewidth=0.7)

    # Event markers (on the spread axis).
    ts_np = df["timestamp"].to_numpy()
    def pos(t):
        return int(np.searchsorted(ts_np, np.datetime64(t)))

    klass_color = {"reversion": "green", "divergence": "red", "continuation": "grey"}
    for e in events:
        c = klass_color[e["klass"]]
        ax.scatter(pos(e["det_time"]), e["det_spread"], marker="o", color=c, s=35, zorder=5)
        ax.scatter(pos(e["ext_time"]), e["ext_spread"], marker="^", color=c, s=55, zorder=5)
        ax.scatter(pos(e["res_time"]), e["res_spread"], marker="x", color=c, s=55, zorder=5)

    # X-tick strategy: single day -> hourly; multi-day -> session boundaries.
    unique_days = df["day_fut"].unique()
    title_extra = ""
    if len(unique_days) == 1:
        ts             = pd.Series(df["timestamp"].to_numpy())
        hour_change    = ts.dt.hour.ne(ts.dt.hour.shift())
        hour_positions = np.where(hour_change.to_numpy())[0]
        hour_labels    = [pd.Timestamp(df["timestamp"].iloc[p]).strftime("%H:%M")
                          for p in hour_positions]
        ax.set_xticks(hour_positions)
        ax.set_xticklabels(hour_labels, rotation=0)
        ax.set_xlabel("time (HH:MM)")
        title_extra = f"  {pd.Timestamp(str(int(unique_days[0]))).strftime('%Y-%m-%d')}"
    else:
        day_change    = df["day_fut"].ne(df["day_fut"].shift())
        day_positions = np.where(day_change.to_numpy())[0]
        day_labels    = [pd.Timestamp(str(int(d))).strftime("%Y-%m-%d")
                         for d in df["day_fut"].iloc[day_positions]]
        ax.set_xticks(day_positions)
        ax.set_xticklabels(day_labels, rotation=45, ha="right")
        ax.set_xlabel("session")
        for p in day_positions[1:]:
            ax.axvline(p, color="grey", linewidth=0.3, alpha=0.4)

    # Legend on the primary axis. Proxy artists, one per thing.
    legend_items = [
        Line2D([], [], color="steelblue",  linewidth=0.8, label="spread (%)"),
        Line2D([], [], color="darkorange", linewidth=0.8, label="cash LTP"),
        Line2D([], [], color="black", linestyle="-",  linewidth=0.8,
               label=f"outlier  (µ ± {out_thresh}σ)"),
        Line2D([], [], color="red",   linestyle="--", linewidth=0.8,
               label=f"divergence (µ ± {out_thresh + rev_thresh}σ)"),
        Line2D([], [], color="green", linestyle="--", linewidth=0.8,
               label=f"reversion  (µ ± {out_thresh - rev_thresh}σ)"),
        Line2D([], [], marker="o", color="grey", linestyle="", label="detection"),
        Line2D([], [], marker="^", color="grey", linestyle="", label="extremum"),
        Line2D([], [], marker="x", color="grey", linestyle="", label="resolution"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(f"{symbol} {contract}{title_extra}   out={out_thresh}   rev={rev_thresh}")
    ax.set_ylabel("spread (%)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_equity_curve(combined):
    equity = combined["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(combined["det_timestamp"], equity, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title("cumulative PnL (spread-% units)")
    ax.set_xlabel("detection time"); ax.set_ylabel("cumulative PnL")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig, ax
