"""Resume las evaluaciones sobre prueba (ya hechas por scripts/evaluar_modelo.py y
scripts/calcular_mof.py) en tablas para el informe. Solo lee archivos de resultados:
no carga modelos, no vuelve a evaluar y no decide nada.

Escribe reports/resultados_prueba.json y, si se pide, fragmentos LaTeX con coma decimal.
La desviacion es la muestral (ddof=1) entre las cinco semillas.

Uso:
    python3 scripts/tablas_resultados_prueba.py --tex <carpeta de salida de los .tex>
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
METRICAS = [
    ("accuracy_global", "Accuracy"),
    ("precision_macro", "\\textit{Precision} macro"),
    ("recall_macro", "\\textit{Recall} macro"),
    ("f1_macro", "F1 macro"),
    ("auc_macro_ovr", "AUC macro"),
]


def cargar():
    datos = {}
    for modelo in ("m1", "m4"):
        for ruta in sorted(glob.glob(str(PROJECT_ROOT / "runs" / "evaluacion" / f"{modelo}_seed*_test_*" / "metricas.json"))):
            semilla = int(Path(ruta).parent.name.split("seed")[1].split("_")[0])
            with open(ruta, "r", encoding="utf-8") as f:
                datos.setdefault(modelo, {})[semilla] = json.load(f)
    for modelo, por_semilla in datos.items():
        if len(por_semilla) != 5 or any(m["n_muestras"] != 3808 for m in por_semilla.values()):
            raise SystemExit(f"{modelo}: se esperaban 5 semillas con 3.808 imagenes; hay {len(por_semilla)}")
    return datos


def ms(valores):
    return st.mean(valores), st.stdev(valores)


def coma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def celda(m, s, d=3):
    return f"{coma(m, d)} $\\pm$ {coma(s, d)}"


def fragmento(filas):
    """Une las filas y cierra con el \\hline final, que va dentro del fragmento para no depender del \\input."""
    return "\n".join(filas) + "\n        \\hline\n"


def main():
    parser = argparse.ArgumentParser(description="Tablas de resultados sobre prueba")
    parser.add_argument("--tex", type=Path, default=None, help="carpeta donde escribir los fragmentos .tex")
    args = parser.parse_args()

    datos = cargar()
    mof = json.load(open(PROJECT_ROOT / "reports" / "mof_resultados.json", "r", encoding="utf-8"))
    semillas = sorted(datos["m1"])

    resumen = {"semillas": semillas, "n_muestras": 3808}

    # Tabla 1: MOF por semilla
    principal = mof["comparacion_principal_mof_ahp"]
    resumen["mof_ahp"] = principal

    # Tabla 2: metricas globales
    glob_ = {}
    for clave, _ in METRICAS:
        a = [datos["m1"][s][clave] for s in semillas]
        b = [datos["m4"][s][clave] for s in semillas]
        glob_[clave] = {"m1": ms(a), "m4": ms(b), "diferencia_media": st.mean(x - y for x, y in zip(a, b)),
                        "m1_mayor_en": sum(x > y for x, y in zip(a, b))}
    resumen["metricas_globales"] = glob_

    # Tabla 3: por clase
    por_clase = {}
    for clave in ("recall_per_class", "precision_per_class", "f1_per_class", "auc_per_class"):
        por_clase[clave] = {c: {m: ms([datos[m][s][clave][c] for s in semillas]) for m in ("m1", "m4")} for c in CLASES}
    resumen["por_clase"] = por_clase

    # Tabla 4: sensibilidad
    V = mof["vectores"]
    sens = {}
    for nombre, v in V.items():
        difs = [f["diferencia"] for f in v["por_semilla"].values()]
        sens[nombre] = {"tipo": v["tipo"], "pesos": v["pesos"], "media": st.mean(difs), "min": min(difs), "max": max(difs),
                        "positivas": sum(d > mof["tolerancia_empate"] for d in difs)}
    resumen["sensibilidad"] = sens

    salida = PROJECT_ROOT / "reports" / "resultados_prueba.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"Escrito {salida}")

    if args.tex is None:
        return
    args.tex.mkdir(parents=True, exist_ok=True)

    # --- MOF por semilla
    filas = []
    for s in semillas:
        p = principal["pares"][str(s)]
        filas.append(f"        {s} & {coma(p['m1'], 4)} & {coma(p['m4'], 4)} & {coma(p['diferencia'], 4)} \\\\")
    filas.append("        \\hline")
    mm1 = st.mean(principal["pares"][str(s)]["m1"] for s in semillas)
    mm4 = st.mean(principal["pares"][str(s)]["m4"] for s in semillas)
    filas.append(f"        Media & {coma(mm1, 4)} & {coma(mm4, 4)} & {coma(principal['media_diferencia'], 4)} \\\\")
    filas.append(f"        Mediana de la diferencia & & & {coma(principal['mediana_diferencia'], 4)} \\\\")
    (args.tex / "tabla_mof_semilla.tex").write_text(fragmento(filas), encoding="utf-8")

    # --- metricas globales
    filas = []
    for clave, etiqueta in METRICAS:
        g = glob_[clave]
        filas.append(f"        {etiqueta} & {celda(*g['m1'])} & {celda(*g['m4'])} & {coma(g['diferencia_media'])} & {g['m1_mayor_en']} de 5 \\\\")
    (args.tex / "tabla_metricas_globales.tex").write_text(fragmento(filas), encoding="utf-8")

    # --- por clase
    filas = []
    for c in CLASES:
        partes = [c]
        for clave in ("recall_per_class", "precision_per_class", "f1_per_class", "auc_per_class"):
            for m in ("m1", "m4"):
                partes.append(coma(por_clase[clave][c][m][0], 2))
        filas.append("        " + " & ".join(partes) + " \\\\")
    (args.tex / "tabla_por_clase.tex").write_text(fragmento(filas), encoding="utf-8")

    # --- sensibilidad
    filas = []
    etiquetas = {"AHP": "AHP (referencia)", "C0": "C0 (pesos iguales)", "C1": "C1 (prioriza \\textit{Accuracy})",
                 "C2": "C2 (prioriza \\textit{Recall})", "C3": "C3 (prioriza AUC)"}
    for nombre in ("AHP", "C0", "C1", "C2", "C3"):
        d = sens[nombre]
        filas.append(f"        {etiquetas[nombre]} & {coma(d['media'], 3)} & {coma(d['min'], 3)} & {d['positivas']} de 5 \\\\")
    filas.append("        \\hline")
    for nombre, texto in (("AHP:sin_accuracy_global", "Ablación: sin \\textit{Accuracy}"),
                          ("AHP:sin_recall_macro", "Ablación: sin \\textit{Recall} macro"),
                          ("AHP:sin_auc_macro_ovr", "Ablación: sin AUC macro")):
        d = sens[nombre]
        filas.append(f"        {texto} & {coma(d['media'], 3)} & {coma(d['min'], 3)} & {d['positivas']} de 5 \\\\")
    pert = [d for n, d in sens.items() if d["tipo"] == "perturbacion"]
    filas.append("        \\hline")
    filas.append(f"        {len(pert)} perturbaciones de $\\pm$10\\,\\% (rango) & {coma(min(d['media'] for d in pert), 3)} a {coma(max(d['media'] for d in pert), 3)} & "
                 f"{coma(min(d['min'] for d in pert), 3)} & {min(d['positivas'] for d in pert)} de 5 \\\\")
    (args.tex / "tabla_sensibilidad.tex").write_text(fragmento(filas), encoding="utf-8")
    print(f"Fragmentos LaTeX en {args.tex}")


if __name__ == "__main__":
    main()
