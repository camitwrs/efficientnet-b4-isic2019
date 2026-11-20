"""Descompone el resultado de melanoma (MEL) de M1 y M4 sobre prueba, a partir de las
predicciones ya guardadas por scripts/evaluar_modelo.py. Es descriptivo: no vuelve a
evaluar ningun modelo ni decide nada con el conjunto de prueba.

Escribe dos fragmentos LaTeX con coma decimal: tabla_melanoma.tex (destino de los errores)
y tabla_melanoma_umbral.tex (medidas que no dependen del umbral, por semilla), e imprime los
intervalos de confianza de la diferencia de Recall de melanoma.

Uso:
    python scripts/analisis_melanoma.py --tex <carpeta de salida>
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

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


def cargar(modelo):
    rutas = sorted(glob.glob(str(PROJECT_ROOT / "runs" / "evaluacion" / f"{modelo}_seed*_test_*" / "predicciones.csv")))
    return {int(Path(r).parent.name.split("seed")[1].split("_")[0]): pd.read_csv(r) for r in rutas}


def precision_en_recall(y, s, r):
    """Mayor Precision entre los umbrales con Recall >= r."""
    p, rc, _ = precision_recall_curve(y, s)
    return p[rc >= r].max()


def recall_con_especificidad(y, s, esp):
    """Mayor Recall entre los umbrales con tasa de falsos positivos <= 1 - esp."""
    fpr, tpr, _ = roc_curve(y, s)
    return tpr[fpr <= 1 - esp].max()


def medidas_sin_umbral(tex):
    a, b = cargar("m1"), cargar("m4")
    semillas = sorted(a)
    medidas = [
        ("AUC de melanoma (uno contra el resto)", lambda y, s: roc_auc_score(y, s)),
        ("\\textit{Recall} con especificidad de 90\\,\\%", lambda y, s: recall_con_especificidad(y, s, 0.90)),
        ("\\textit{Recall} con especificidad de 80\\,\\%", lambda y, s: recall_con_especificidad(y, s, 0.80)),
    ]
    filas = []
    for nombre, f in medidas:
        v = {m: np.array([f((d[s].clase_verdadera == MEL).values, d[s].prob_MEL.values) for s in semillas]) for m, d in (("m1", a), ("m4", b))}
        celda = lambda x: f"{coma(x.mean(), 3)} $\\pm$ {coma(x.std(ddof=1), 3)}"
        filas.append(f"        {nombre} & {celda(v['m1'])} & {celda(v['m4'])} & {int((v['m1'] > v['m4']).sum())} de 5 \\\\")
    (tex / "tabla_melanoma_umbral.tex").write_text("\n".join(filas) + "\n        \\hline\n", encoding="utf-8")
    print("\n".join(filas))
    # Intervalos de la diferencia de Recall de melanoma (M1 menos M4)
    rng = np.random.default_rng(0)
    mel = a[semillas[0]].index[a[semillas[0]].clase_verdadera == MEL]
    A = np.array([(a[s].loc[mel].clase_predicha == MEL).values for s in semillas]).astype(float)
    B = np.array([(b[s].loc[mel].clase_predicha == MEL).values for s in semillas]).astype(float)
    n = len(mel)
    bs = [A[:, i].mean() - B[:, i].mean() for i in (rng.integers(0, n, n) for _ in range(4000))]
    d = A.mean(1) - B.mean(1)
    t = 2.776  # t de Student, 4 grados de libertad, 95 %
    se = d.std(ddof=1) / np.sqrt(len(d))
    y = (a[semillas[0]].clase_verdadera == MEL).values
    sa = np.array([a[k].prob_MEL.values for k in semillas])
    sb = np.array([b[k].prob_MEL.values for k in semillas])
    for esp in (0.90, 0.80):
        dd = np.array([recall_con_especificidad(y, sa[k], esp) - recall_con_especificidad(y, sb[k], esp) for k in range(len(semillas))])
        sed = dd.std(ddof=1) / np.sqrt(len(dd))
        bb = []
        for _ in range(600):
            idx = rng.integers(0, len(y), len(y))
            bb.append(np.mean([recall_con_especificidad(y[idx], sa[k][idx], esp) - recall_con_especificidad(y[idx], sb[k][idx], esp)
                               for k in range(len(semillas))]))
        print(f"Recall a especificidad {esp:.0%}: diferencia {dd.mean():+.3f}; IC95 % por semillas (t) {dd.mean() - t * sed:+.3f} a "
              f"{dd.mean() + t * sed:+.3f}; IC95 % por muestreo de imagenes (bootstrap) {np.percentile(bb, 2.5):+.3f} a {np.percentile(bb, 97.5):+.3f}")
    print(f"diferencia de Recall MEL {A.mean() - B.mean():+.4f}; IC95 % por muestreo de imagenes (bootstrap) "
          f"{np.percentile(bs, 2.5):+.3f} a {np.percentile(bs, 97.5):+.3f}; IC95 % por semillas (t, n=5) "
          f"{d.mean() - t * se:+.3f} a {d.mean() + t * se:+.3f}")


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
    medidas_sin_umbral(a.tex)


if __name__ == "__main__":
    main()
