"""Comparacion pareada M1 vs M4 por semilla, segun el protocolo declarado en el
informe (Capitulo 5, comparacion principal) antes de consultar el test.

Para cada semilla se toma la diferencia d = valor(M1) - valor(M4). Se reportan
los pares, las diferencias, su media y mediana, el numero efectivo de pares y,
como contraste complementario, Wilcoxon de rangos con signo exacto:

- las diferencias se redondean a 12 decimales antes de asignar rangos;
- los ceros se excluyen del estadistico y se reportan aparte;
- los empates en magnitud reciben rangos medios;
- el valor p bilateral se obtiene enumerando los 2^n patrones de signos de las
  n diferencias no nulas, con el estadistico de extremidad |T+ - S/2|, donde T+
  es la suma de rangos positivos y S la suma de todos los rangos;
- no se usa aproximacion normal.

El tamano del efecto es la correlacion biserial por rangos r = (T+ - T-) / S,
positiva a favor de M1. Con cinco pares no nulos y sin empates el p minimo es
2/2^5 = 0,0625, por eso el analisis es descriptivo.
"""

from itertools import product
from statistics import mean, median

DECIMALES = 12
# Holgura para comparar extremidades calculadas con rangos medios (sumas de medios).
_EPS = 1e-9


def _rangos_medios(valores):
    """Rangos 1..n de los valores, con rango medio para los empates."""
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    rangos = [0.0] * len(valores)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and valores[orden[j + 1]] == valores[orden[i]]:
            j += 1
        medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rangos[orden[k]] = medio
        i = j + 1
    return rangos


def wilcoxon_exacto(diferencias):
    """Wilcoxon de rangos con signo, exacto y bilateral, sobre las diferencias dadas.

    Devuelve un diccionario con n_efectivo, n_ceros, t_mas, t_menos, suma_rangos,
    p_bilateral y r_biserial. Si todas las diferencias son cero, p_bilateral y
    r_biserial son None: no se aplica el contraste.
    """
    redondeadas = [round(float(d), DECIMALES) for d in diferencias]
    no_nulas = [d for d in redondeadas if d != 0.0]
    n_ceros = len(redondeadas) - len(no_nulas)
    if not no_nulas:
        return {"n_efectivo": 0, "n_ceros": n_ceros, "t_mas": None, "t_menos": None,
                "suma_rangos": 0.0, "p_bilateral": None, "r_biserial": None}

    rangos = _rangos_medios([abs(d) for d in no_nulas])
    suma = sum(rangos)
    t_mas = sum(r for r, d in zip(rangos, no_nulas) if d > 0)
    t_menos = suma - t_mas
    extremidad_obs = abs(t_mas - suma / 2)

    extremos = 0
    patrones = 0
    for signos in product((1, -1), repeat=len(rangos)):
        t = sum(r for r, s in zip(rangos, signos) if s > 0)
        patrones += 1
        if abs(t - suma / 2) >= extremidad_obs - _EPS:
            extremos += 1

    return {
        "n_efectivo": len(no_nulas),
        "n_ceros": n_ceros,
        "t_mas": t_mas,
        "t_menos": t_menos,
        "suma_rangos": suma,
        "p_bilateral": extremos / patrones,
        "r_biserial": (t_mas - t_menos) / suma,
    }


def comparacion_pareada(valores_m1, valores_m4):
    """Compara dos diccionarios {semilla: valor} con las mismas semillas.

    Devuelve los pares, las diferencias M1 - M4, su media y mediana, y el
    resultado de wilcoxon_exacto sobre esas diferencias.
    """
    if set(valores_m1) != set(valores_m4):
        raise ValueError(f"Las semillas de M1 {sorted(valores_m1)} y M4 {sorted(valores_m4)} no coinciden")
    semillas = sorted(valores_m1)
    diferencias = {s: valores_m1[s] - valores_m4[s] for s in semillas}
    return {
        "pares": {s: {"m1": valores_m1[s], "m4": valores_m4[s], "diferencia": diferencias[s]} for s in semillas},
        "media_diferencia": mean(diferencias.values()),
        "mediana_diferencia": median(diferencias.values()),
        "wilcoxon": wilcoxon_exacto([diferencias[s] for s in semillas]),
    }
