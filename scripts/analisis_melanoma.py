"""Descompone el resultado de melanoma (MEL) de M1 y M4 sobre prueba, a partir de las
predicciones ya guardadas por scripts/evaluar_modelo.py. Es descriptivo: no vuelve a
evaluar ningun modelo ni decide nada con el conjunto de prueba.

Escribe un fragmento LaTeX con coma decimal (tabla_melanoma.tex).

Uso:
    python scripts/analisis_melanoma.py --tex <carpeta de salida>
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
MEL = CLASES.index("MEL")


def coma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def resumen(modelo):
    filas, predichos, recall, precision = [], [], [], []
    rutas = sorted(glob.glob(str(PROJECT_ROOT / "runs" / "evaluacion" / f"{modelo}_seed*_test_*" / "predicciones.csv")))
    if len(rutas) != 5:
        raise SystemExit(f"{modelo}: se esperaban 5 evaluaciones sobre prueba; hay {len(rutas)}")
    for ruta in rutas:
        d = pd.read_csv(ruta)
        mel = d[d.clase_verdadera == MEL]
        filas.append(mel.clase_predicha.value_counts().reindex(range(8), fill_value=0).values)
        pm = (d.clase_predicha == MEL).sum()
        predichos.append(pm)
        recall.append((mel.clase_predicha == MEL).mean())
        precision.append(((d.clase_predicha == MEL) & (d.clase_verdadera == MEL)).sum() / pm)
    return {"destino": np.mean(filas, axis=0), "predichos": np.mean(predichos), "recall": recall, "precision": precision,
            "n_mel": int(np.sum(filas[0]))}


def main():
    p = argparse.ArgumentParser(description="Descomposicion del resultado de melanoma")
    p.add_argument("--tex", type=Path, required=True)
    a = p.parse_args()
    r = {m: resumen(m) for m in ("m1", "m4")}
    a.tex.mkdir(parents=True, exist_ok=True)
    fila = lambda nombre, f: f"        {nombre} & " + " & ".join(f(r[m]) for m in ("m1", "m4")) + " \\\\"
    otras = lambda x: x["destino"][[0, 1, 3, 6, 7]].sum()
    filas = [
        fila("Melanomas en la prueba", lambda x: str(x["n_mel"])),
        fila("Melanomas detectados", lambda x: f"{x['destino'][MEL]:.0f}"),
        fila("Imágenes predichas como melanoma", lambda x: f"{x['predichos']:.0f}"),
        fila("\\textit{Precision} de melanoma (media por semilla)", lambda x: coma(np.mean(x["precision"]))),
        fila("\\textit{Recall} de melanoma, rango entre semillas", lambda x: f"{coma(min(x['recall']))} a {coma(max(x['recall']))}"),
        fila("Melanomas perdidos hacia nevo", lambda x: f"{x['destino'][CLASES.index('NV')]:.0f}"),
        fila("Melanomas perdidos hacia lesión queratósica benigna", lambda x: f"{x['destino'][CLASES.index('BKL')]:.0f}"),
        fila("Melanomas perdidos hacia otras clases", lambda x: f"{otras(x):.0f}"),
    ]
    (a.tex / "tabla_melanoma.tex").write_text("\n".join(filas) + "\n        \\hline\n", encoding="utf-8")
    print("\n".join(filas))


if __name__ == "__main__":
    main()
