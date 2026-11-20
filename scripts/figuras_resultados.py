"""Figuras de resultados del informe (PDF vectorial), a partir de los archivos de resultados.
Solo lee reports/mof_resultados.json y reports/resultados_prueba.json: no carga modelos ni
vuelve a evaluar.

Uso:
    python scripts/figuras_resultados.py --salida <carpeta>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
C1, C4 = "#294C60", "#C4731B"  # M1 azul, M4 naranja, fijos en todas las figuras
CLASES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def coma(ax, eje="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def mof_pareada(mof, salida):
    pares = mof["comparacion_principal_mof_ahp"]["pares"]
    semillas = sorted(pares, key=int)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, s in enumerate(semillas):
        a, b = pares[s]["m1"], pares[s]["m4"]
        ax.plot([0, 1], [b, a], color="0.65", lw=1, zorder=1)
        ax.scatter([0], [b], color=C4, zorder=2, s=28)
        ax.scatter([1], [a], color=C1, zorder=2, s=28)
        ax.text(-0.05, b, f"semilla {s}", va="center", ha="right", fontsize=7, color="0.35")
    ax.set_xticks([0, 1], ["M4", "M1"])
    ax.set_xlim(-0.55, 1.2)
    ax.set_ylabel("MOF con pesos AHP")
    ax.set_ylim(0.35, 0.75)
    coma(ax)
    ax.grid(axis="y", lw=0.3)
    fig.tight_layout()
    fig.savefig(salida / "mof_pareada.pdf")
    plt.close(fig)


def recall_por_clase(res, salida):
    r = res["por_clase"]["recall_per_class"]
    y = np.arange(len(CLASES))[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for dy, m, col, nombre in ((0.15, "m1", C1, "M1"), (-0.15, "m4", C4, "M4")):
        med = [r[c][m][0] for c in CLASES]
        sd = [r[c][m][1] for c in CLASES]
        ax.errorbar(med, y + dy, xerr=sd, fmt="o", color=col, ms=4, capsize=2, lw=1, label=nombre)
    ax.set_yticks(y, CLASES)
    ax.get_yticklabels()[CLASES.index("MEL")].set_fontweight("bold")
    ax.set_xlabel(r"Recall (media $\pm$ desviación estándar, 5 semillas)")
    ax.set_xlim(0, 1)
    coma(ax, "x")
    ax.grid(axis="x", lw=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(salida / "recall_por_clase.pdf")
    plt.close(fig)


def simplex(mof, res, salida):
    g = res["metricas_globales"]
    d = np.array([g["accuracy_global"]["diferencia_media"], g["recall_macro"]["diferencia_media"],
                  g["auc_macro_ovr"]["diferencia_media"]])
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])  # Accuracy, Recall, AUC
    n = 200
    pts, val = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            w = np.array([i, j, n - i - j]) / n
            pts.append(w @ V)
            val.append(w @ d)
    pts, val = np.array(pts), np.array(val)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sc = ax.tricontourf(pts[:, 0], pts[:, 1], val, levels=np.linspace(val.min(), val.max(), 12), cmap="Blues")
    ax.plot(*np.vstack([V, V[0]]).T, color="0.2", lw=0.8)
    for (x, y), t in zip(V, ["Accuracy\n" + f"{d[0]:.3f}".replace(".", ","), "Recall macro\n" + f"{d[1]:.3f}".replace(".", ","),
                             "AUC macro\n" + f"{d[2]:.3f}".replace(".", ",")]):
        ax.text(x, y + (0.07 if y > 0 else -0.09), t, ha="center", va="center", fontsize=8)
    marcas = {"AHP": ("AHP", "o"), "C0": ("C0", "s"), "C1": ("C1", "s"), "C2": ("C2", "s"), "C3": ("C3", "s")}
    for k, (t, mk) in marcas.items():
        w = mof["vectores"][k]["pesos"]
        p = np.array([w["accuracy_global"], w["recall_macro"], w["auc_macro_ovr"]]) @ V
        ax.scatter(*p, marker=mk, color=C4 if k == "AHP" else "0.15", s=34, zorder=3)
        ax.annotate(t, p, xytext=(5, 4), textcoords="offset points", fontsize=7)
    for k, t in (("AHP:sin_accuracy_global", "sin Acc."), ("AHP:sin_recall_macro", "sin Rec."), ("AHP:sin_auc_macro_ovr", "sin AUC")):
        w = mof["vectores"][k]["pesos"]
        p = np.array([w["accuracy_global"], w["recall_macro"], w["auc_macro_ovr"]]) @ V
        ax.scatter(*p, marker="^", color="0.15", s=26, zorder=3)
        ax.annotate(t, p, xytext=(4, -9), textcoords="offset points", fontsize=7)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Diferencia MOF (M1 $-$ M4), media de 5 semillas", fontsize=8)
    cb.formatter = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    cb.update_ticks()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(salida / "simplex_pesos.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    mof = json.load(open(PROJECT_ROOT / "reports" / "mof_resultados.json", encoding="utf-8"))
    res = json.load(open(PROJECT_ROOT / "reports" / "resultados_prueba.json", encoding="utf-8"))
    mof_pareada(mof, a.salida)
    recall_por_clase(res, a.salida)
    simplex(mof, res, a.salida)
    print("Figuras en", a.salida)


if __name__ == "__main__":
    main()
