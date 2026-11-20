"""Curvas de val_loss por epoca de las diez corridas finales (solo entrenamiento y validacion).
Lee los history.csv de runs/final/; no usa el conjunto de prueba.

Uso:
    python scripts/curvas_entrenamiento.py --salida <carpeta>
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL = PROJECT_ROOT / "runs" / "final"
SEMILLAS = [42, 123, 456, 789, 2026]
C1, C4 = "#294C60", "#C4731B"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def leer(patron):
    ruta = sorted(glob.glob(str(FINAL / patron)))
    if len(ruta) != 1:
        raise SystemExit(f"se esperaba una carpeta para {patron}; hay {len(ruta)}")
    return pd.read_csv(Path(ruta[0]) / "history.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for s in SEMILLAS:
        f1 = leer(f"m1_fase1/seed{s}_*")
        f2 = leer(f"m1_fase2/block3_top_seed{s}_*")
        y = list(f1["val_loss"]) + list(f2["val_loss"])
        x = list(range(1, len(y) + 1))
        ejes[0].plot(x, y, color=C1, lw=0.9, alpha=0.8)
        ejes[0].axvline(len(f1) + 0.5, color=C1, lw=0.4, ls=":", alpha=0.6)
        k = int(pd.Series(y).idxmin())
        ejes[0].scatter([x[k]], [y[k]], color=C1, s=16, zorder=3)
        m4 = leer(f"m4/seed{s}_*")
        y4 = list(m4["val_loss"])
        ejes[1].plot(range(1, len(y4) + 1), y4, color=C4, lw=0.9, alpha=0.8)
        k4 = int(pd.Series(y4).idxmin())
        ejes[1].scatter([k4 + 1], [y4[k4]], color=C4, s=16, zorder=3)
    ejes[0].set_title("M1 (Fase 1 y Fase 2; línea punteada: cambio de fase)", fontsize=8)
    ejes[1].set_title("M4", fontsize=8)
    for e in ejes:
        e.set_xlabel("Época")
        e.grid(lw=0.3)
    ejes[0].set_ylabel("val_loss")
    ejes[0].set_ylim(1.1, 2.4)
    fig.tight_layout()
    fig.savefig(a.salida / "curvas_val_loss.pdf")
    print("Figura en", a.salida)


if __name__ == "__main__":
    main()
