import numpy as np
import matplotlib.pyplot as plt

def plot_dispatch(ts, title="(a) Electric power dispatch"):
    """ts: dict[str, list[float]]，键包含 Wind/PV/CHP/ES/ER/ED"""
    T = len(ts["ED"])
    x = np.arange(1, T+1)

    # 正负分开堆叠：ES/ER 可能为负（充电/售电），为了视觉一致，分成正柱与负柱分别画
    def pos(a): return np.clip(np.asarray(a, dtype=float), 0, None)
    def neg(a): return np.clip(np.asarray(a, dtype=float), None, 0)

    wind = np.asarray(ts["Wind"], dtype=float)
    pv   = np.asarray(ts["PV"], dtype=float)
    chp  = np.asarray(ts["CHP"], dtype=float)
    es_p, es_n = pos(ts["ES"]), neg(ts["ES"])
    er_p, er_n = pos(ts["ER"]), neg(ts["ER"])
    ed   = np.asarray(ts["ED"], dtype=float)

    # 先画正向功率堆叠（Wind, PV, CHP, ES+, ER+）
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(T)
    bars = []
    for name, arr in [("Wind", wind), ("PV", pv), ("CHP", chp), ("ES", es_p), ("ER", er_p)]:
        b = ax.bar(x, arr, bottom=bottom, width=0.9, label=name, alpha=0.85, linewidth=0.2, edgecolor="k")
        bars.append(b)
        bottom += arr

    # 再画负向功率堆叠（ES−, ER−）
    bottom = np.zeros(T)
    for name, arr in [("ES-", -es_n), ("ER-", -er_n)]:  # 取绝对值向下画
        ax.bar(x, -arr, bottom=-bottom, width=0.9, label=name, alpha=0.85, linewidth=0.2, edgecolor="k")
        bottom += arr

    # 需求折线
    ax.plot(x, ed, "k.-", linewidth=1.8, markersize=3, label="ED")

    ax.set_xlim(0.5, T+0.5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(ncol=6, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()
