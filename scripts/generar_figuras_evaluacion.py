"""Dibuja la matriz de confusion y las curvas ROC por clase de evaluaciones ya
hechas por scripts/evaluar_modelo.py. Solo lee predicciones.csv: no carga
modelos, no recalcula metricas para decidir nada y no toca otros conjuntos.

Por cada carpeta de evaluacion guarda en <carpeta>/figuras/:
- matriz_confusion.pdf/png: filas = clase verdadera, normalizada por fila
  (recall por clase en la diagonal), con el conteo en cada celda;
- roc_por_clase.pdf/png: curva ROC uno contra el resto de cada clase con su AUC
  y la curva macro (promedio de las TPR interpoladas sobre una grilla comun de FPR).

Uso:
    python scripts/generar_figuras_evaluacion.py runs/evaluacion/m1_seed42_test_* runs/evaluacion/m4_seed42_test_*
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def leer(carpeta):
    pred = pd.read_csv(carpeta / "predicciones.csv")
    columnas = [c for c in pred.columns if c.startswith("prob_")]
    clases = [c[len("prob_"):] for c in columnas]
    with open(carpeta / "evaluacion_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return pred["clase_verdadera"].to_numpy(), pred[columnas].to_numpy(), clases, info


# Texto de las figuras en espanol, con tildes, porque van al informe.
PARTICIONES = {"test": "prueba", "validation": "validación"}


def titulo(info):
    return f"{info['modelo'].upper()} | semilla {info['seed']} | {PARTICIONES.get(info['particion'], info['particion'])}"


def guardar(fig, destino, nombre):
    destino.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(destino / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def matriz_confusion(y_true, y_prob, clases, info, destino):
    cm = confusion_matrix(y_true, y_prob.argmax(axis=1), labels=list(range(len(clases))))
    normalizada = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if normalizada[i, j] > 0.5 else "black"
            ax.text(j, i, f"{normalizada[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(clases)), clases)
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase verdadera")
    ax.set_title(f"Matriz de confusión normalizada por fila\n{titulo(info)}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    guardar(fig, destino, "matriz_confusion")


def curvas_roc(y_true, y_prob, clases, info, destino):
    grilla = np.linspace(0, 1, 1001)
    tprs = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, clase in enumerate(clases):
        fpr, tpr, _ = roc_curve((y_true == k).astype(int), y_prob[:, k])
        tprs.append(np.interp(grilla, fpr, tpr))
        ax.plot(fpr, tpr, lw=1, label=f"{clase} (AUC = {auc(fpr, tpr):.3f})")
    tpr_macro = np.mean(tprs, axis=0)
    tpr_macro[0] = 0.0
    ax.plot(grilla, tpr_macro, color="black", lw=2, linestyle="--",
            label=f"promedio macro de las curvas (AUC = {auc(grilla, tpr_macro):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC uno contra el resto\n{titulo(info)}")
    ax.legend(loc="lower right", fontsize=8)
    guardar(fig, destino, "roc_por_clase")


def main():
    parser = argparse.ArgumentParser(description="Figuras de matriz de confusion y ROC desde predicciones.csv")
    parser.add_argument("carpetas", nargs="+", type=Path, help="carpetas creadas por evaluar_modelo.py")
    args = parser.parse_args()
    for carpeta in args.carpetas:
        y_true, y_prob, clases, info = leer(carpeta)
        destino = carpeta / "figuras"
        matriz_confusion(y_true, y_prob, clases, info, destino)
        curvas_roc(y_true, y_prob, clases, info, destino)
        print(f"Figuras guardadas en {destino}")


if __name__ == "__main__":
    main()
