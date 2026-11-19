"""Diagnostico exploratorio del artefacto de esquina observado en los mapas Grad-CAM.

Al revisar las figuras de M1 (semilla 42) aparecio una celda caliente en la esquina superior
derecha, independiente del contenido de la imagen. Hipotesis: es un efecto de borde de la
red (relleno de las convoluciones), no evidencia de la lesion. Este script la pone a prueba
sobre las mismas imagenes que selecciono la regla declarada, sin elegir nada nuevo:

1. Posicion de la celda maxima del mapa crudo (a la resolucion de la capa) y cuanto vale la
   celda de cada esquina respecto del maximo, sobre las imagenes seleccionadas.
2. Reflejo horizontal y vertical: si la celda caliente se queda en la misma esquina del
   mapa aunque la imagen este reflejada, depende de la posicion y no del contenido.
3. Imagen gris constante: sin contenido, cualquier estructura espacial viene de la red.
4. Activacion media por celda del mapa de la capa objetivo, para ver si el borde ya tiene
   activaciones distintas antes de ponderar por el gradiente.

Es un diagnostico posterior a ver las figuras: no cambia la seleccion, las metricas ni los
mapas del informe, y asi se declara. Lee el conjunto de test solo para las mismas imagenes
que ya visualizo scripts/generar_gradcam.py.

Uso:
    python scripts/diagnostico_gradcam_esquina.py --evaluacion runs/evaluacion/m1_seed42_test_<fecha> --confirmo-config-congelada
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from scripts.evaluar_modelo import construir_modelo, sha256_de
from scripts.train_m1_fase1 import CONFIG_PATH, PARTITION_CSV, load_config
from src.data.isic2019 import load_image
from src.evaluation.gradcam import calcular_gradcam, capa_objetivo, seleccionar_imagenes

LOTE = 8


def crudos(model, capa, x, clases):
    """Mapas crudos (N, h, w) de Grad-CAM, en lotes."""
    salida = []
    for i in range(0, len(x), LOTE):
        _, crudo = calcular_gradcam(model, capa, x[i:i + LOTE], clases[i:i + LOTE], x.shape[1])
        salida.append(crudo)
    return np.concatenate(salida)


def activaciones(model, nombre_capa, x):
    """Mapa de activaciones A de la capa objetivo, (N, h, w, K), avanzando capa por capa."""
    capas = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    salida = []
    for i in range(0, len(x), LOTE):
        h = tf.convert_to_tensor(x[i:i + LOTE], dtype=tf.float32)
        for capa in capas[:-1]:
            h = capa(h, training=False)
            if capa.name == nombre_capa:
                salida.append(h.numpy())
                break
    return np.concatenate(salida)


def estadisticas(mapas):
    """Resumen de posicion del maximo y de las esquinas, sobre mapas crudos (N, h, w)."""
    n, h, w = mapas.shape
    maximo = mapas.reshape(n, -1).max(axis=1)
    util = maximo > 0
    m = mapas[util] / maximo[util][:, None, None]
    argmax = np.array([np.unravel_index(a.argmax(), a.shape) for a in mapas[util]])
    esquinas = {"sup_izq": (0, 0), "sup_der": (0, w - 1), "inf_izq": (h - 1, 0), "inf_der": (h - 1, w - 1)}
    k = 2  # bloque de k x k celdas en cada esquina
    bloques = {"sup_izq": m[:, :k, :k], "sup_der": m[:, :k, w - k:], "inf_izq": m[:, h - k:, :k], "inf_der": m[:, h - k:, w - k:]}
    return {
        "n_mapas": int(n),
        "n_mapas_no_nulos": int(util.sum()),
        "forma_mapa": [int(h), int(w)],
        "fraccion_maximo_en_bloque_2x2_de_esquina": {
            nombre: float(np.mean([(f < k or f >= h - k) and (c < k or c >= w - k) and
                                   ((f < k) == (nombre.startswith("sup"))) and ((c < k) == (nombre.endswith("izq")))
                                   for f, c in argmax])) for nombre in esquinas},
        "esperado_por_azar_bloque_2x2": float(k * k / (h * w)),
        "valor_medio_celda_esquina_sobre_maximo": {nombre: float(m[:, f, c].mean()) for nombre, (f, c) in esquinas.items()},
        "valor_medio_bloque_2x2_sobre_maximo": {nombre: float(b.mean()) for nombre, b in bloques.items()},
        "valor_medio_centro_sobre_maximo": float(m[:, h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1].mean()),
        "mapa_medio_normalizado": m.mean(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostico del artefacto de esquina en Grad-CAM")
    parser.add_argument("--evaluacion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--confirmo-config-congelada", action="store_true")
    args = parser.parse_args()

    with open(args.evaluacion / "evaluacion_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    if info["particion"] == "test" and not args.confirmo_config_congelada:
        raise SystemExit("La evaluacion es sobre test: confirma con --confirmo-config-congelada.")

    config = load_config(CONFIG_PATH)
    reglas = config["evaluation"]["gradcam"]
    class_order = config["dataset"]["classes"]["order"]
    ckpt = args.checkpoint or Path(info["checkpoint"])
    if sha256_de(ckpt) != info["checkpoint_sha256"]:
        raise SystemExit("El SHA-256 del checkpoint no coincide con el de la evaluacion")

    pred = pd.read_csv(args.evaluacion / "predicciones.csv")
    seleccion = seleccionar_imagenes(pred, class_order, reglas["selection"], reglas["selection"]["seed"])
    model, image_size, normalizar = construir_modelo(info["modelo"], config)
    model.load_weights(str(ckpt))
    capa = capa_objetivo(model, info["modelo"])

    rutas = pd.read_csv(PARTITION_CSV).set_index("image")["image_path_relative"]
    originales = np.stack([load_image(str(args.data_root / rutas[n]), image_size).numpy() for n in seleccion["image"]])
    x = np.stack([normalizar(o) for o in originales]).astype("float32")
    clases = np.array([class_order.index(c) for c in seleccion["clase_predicha"]], dtype="int32")

    variantes = {
        "original": x,
        "reflejo_horizontal": x[:, :, ::-1, :],
        "reflejo_vertical": x[:, ::-1, :, :],
    }
    resultado = {"modelo": info["modelo"], "semilla": info["seed"], "capa": capa, "n_imagenes": int(len(x)),
                 "checkpoint_sha256": info["checkpoint_sha256"], "variantes": {}}
    medios = {}
    for nombre, v in variantes.items():
        est = estadisticas(crudos(model, capa, np.ascontiguousarray(v), clases))
        medios[nombre] = np.array(est.pop("mapa_medio_normalizado"))
        resultado["variantes"][nombre] = est

    # Imagen gris constante: sin contenido. Se explica cada una de las ocho clases.
    gris = np.full((len(class_order), image_size, image_size, 3), 128.0, dtype="float32")
    gris = np.stack([normalizar(g) for g in gris]).astype("float32")
    mapas_gris = crudos(model, capa, gris, np.arange(len(class_order), dtype="int32"))
    resultado["imagen_gris"] = {
        clase: {"celda_maxima": [int(i) for i in np.unravel_index(m.argmax(), m.shape)], "maximo": float(m.max())}
        for clase, m in zip(class_order, mapas_gris)}

    # Activacion media por celda (antes de ponderar por el gradiente).
    A = activaciones(model, capa, x)
    act = np.abs(A).mean(axis=(0, 3))
    resultado["activacion_media_abs_por_celda"] = {
        "esquina_sup_der": float(act[0, -1]), "esquina_sup_izq": float(act[0, 0]),
        "esquina_inf_izq": float(act[-1, 0]), "esquina_inf_der": float(act[-1, -1]),
        "centro": float(act[act.shape[0] // 2, act.shape[1] // 2]), "media_global": float(act.mean()),
    }

    salida = args.evaluacion / "gradcam" / "diagnostico_esquina.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    fig, ejes = plt.subplots(1, 5, figsize=(20, 4.2))
    paneles = [("original", medios["original"]), ("reflejo horizontal", medios["reflejo_horizontal"]),
               ("reflejo vertical", medios["reflejo_vertical"]), ("activación media |A|", act / act.max()),
               ("imagen gris (clase 0)", mapas_gris[0] / max(mapas_gris[0].max(), 1e-12))]
    for eje, (titulo, m) in zip(ejes, paneles):
        im = eje.imshow(m, cmap="viridis", vmin=0, vmax=1)
        eje.set_title(titulo, fontsize=10)
        eje.set_xticks([]); eje.set_yticks([])
    fig.suptitle(f"{info['modelo'].upper()} semilla {info['seed']}: mapa crudo medio normalizado por imagen "
                 f"({len(x)} imágenes seleccionadas) y controles", fontsize=11)
    fig.colorbar(im, ax=ejes, fraction=0.012)
    fig.savefig(salida.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: resultado[k] for k in ("modelo", "capa", "n_imagenes")}, ensure_ascii=False))
    for nombre, est in resultado["variantes"].items():
        print(nombre, "| forma", est["forma_mapa"], "| max en bloque 2x2 de esquina:",
              {k: round(v, 3) for k, v in est["fraccion_maximo_en_bloque_2x2_de_esquina"].items()},
              "| azar:", round(est["esperado_por_azar_bloque_2x2"], 3))
        print("   valor medio de la celda de esquina / maximo:",
              {k: round(v, 3) for k, v in est["valor_medio_celda_esquina_sobre_maximo"].items()},
              "| centro:", round(est["valor_medio_centro_sobre_maximo"], 3))
    print("imagen gris, celda maxima por clase:", {c: v["celda_maxima"] for c, v in resultado["imagen_gris"].items()})
    print("activacion media |A|:", {k: round(v, 4) for k, v in resultado["activacion_media_abs_por_celda"].items()})
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
