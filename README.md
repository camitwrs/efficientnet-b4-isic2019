# EfficientNet-B4 frente a una CNN desde cero en ISIC 2019

Comparación controlada de dos clasificadores de lesiones cutáneas sobre las ocho clases del conjunto de entrenamiento de ISIC 2019, con una función multiobjetivo (MOF) post-hoc cuyos pesos se obtuvieron con el Proceso Analítico Jerárquico (AHP).

*English summary:* controlled comparison of EfficientNet-B4 (transfer learning and fine-tuning, M1) against a four-block CNN trained from scratch (M4) on the eight classes of the ISIC 2019 training set, using a lesion-level split, five paired seeds and a post-hoc multi-objective score (accuracy, macro recall, macro AUC) weighted with AHP.

Trabajo del Proyecto de Título, Pontificia Universidad Católica de Valparaíso (PUCV). Autores: Camila Torres Vega y Jan Houter Hube.

## Diseño

| Elemento | Detalle |
|---|---|
| Datos | ISIC 2019, 25.331 imágenes, ocho clases (AK, BCC, BKL, DF, MEL, NV, SCC, VASC) |
| Partición | 70/15/15 por lesión, semilla 42: 17.705 / 3.818 / 3.808 imágenes |
| M1 | EfficientNet-B4 con pesos ImageNet; Fase 1 con el backbone congelado y Fase 2 desde `block3_top` (elegido en validación) |
| M4 | CNN de cuatro bloques (457.704 parámetros), entrenada desde cero |
| Desbalance | Pesos de clase en la pérdida, N / (K n_c) |
| Corridas | Cinco semillas por modelo (42, 123, 456, 789, 2026), diez corridas finales |
| MOF | Accuracy, Recall macro y AUC macro con pesos AHP (0,1047; 0,6370; 0,2583), congelados antes de leer la prueba |

La prueba se leyó una sola vez, con la configuración y los pesos ya congelados. La comparación contrasta dos configuraciones completas: no separa el efecto de la arquitectura del efecto del preentrenamiento.

## Resultado principal (conjunto de prueba, media de cinco semillas)

| | Accuracy | Recall macro | AUC macro | MOF (AHP) |
|---|---|---|---|---|
| M1 | 0,657 | 0,556 | 0,872 | 0,6485 |
| M4 | 0,448 | 0,366 | 0,798 | 0,4862 |

M1 supera a M4 en las cinco semillas (diferencia media de la MOF: 0,1623; Wilcoxon exacto bilateral p = 0,0625, el mínimo posible con cinco pares). En melanoma el Recall es el mismo en ambos modelos (0,38). Los valores absolutos son moderados y no se comparan de forma directa con otros estudios; ver el informe.

## Estructura

```
configs/    configuración congelada de M1 y M4 y matriz AHP (pilot_m1_m4.yaml)
src/        datos, modelos, métricas, MOF, comparación pareada y Grad-CAM
scripts/    entrenamiento, evaluación, MOF, tablas y figuras
jobs/       plantillas SLURM del HPC Océano PUCV
notebooks/  inventario, partición, preprocesamiento y verificación del modelo
tests/      pruebas automatizadas
reports/    inventario, partición, pesos AHP, selección de profundidad y resultados
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

Las corridas finales usaron Python 3.10, TensorFlow 2.16.1 y Keras 3.12.4.

## Datos y archivos no incluidos

- Las imágenes de ISIC 2019 no se redistribuyen. Se obtienen de la fuente original del desafío y se colocan en `data/`. El archivo `reports/splits/isic2019/isic2019_particion_70_15_15_seed42.csv` contiene los identificadores de la partición.
- Los *checkpoints* y las salidas por corrida (`runs/`) pesan demasiado para el repositorio y no se versionan.

## Reproducción

El orden de ejecución, de la partición a las figuras, es:

| Etapa | Script o notebook |
|---|---|
| Inventario y partición | `notebooks/01_datos_isic2019.ipynb`, `notebooks/02_preprocesamiento_y_particiones.ipynb` |
| Pesos AHP | `scripts/compute_ahp.py` |
| Selección de profundidad de M1 | `scripts/seleccionar_descongelado.py` |
| Entrenamiento | `scripts/train_m1_fase1.py`, `scripts/train_m1_fase2.py`, `scripts/train_m4.py` (plantillas en `jobs/`) |
| Evaluación sobre prueba | `scripts/evaluar_modelo.py` |
| MOF y sensibilidad | `scripts/calcular_mof.py` |
| Tablas y figuras | `scripts/tablas_resultados_prueba.py`, `scripts/generar_figuras_evaluacion.py`, `scripts/figuras_resultados.py`, `scripts/curvas_entrenamiento.py` |
| Grad-CAM | `scripts/generar_gradcam.py`, `scripts/diagnostico_gradcam_esquina.py` |

## Licencia

El código se distribuye bajo licencia MIT (ver `LICENSE`). Las imágenes de ISIC 2019 tienen su propia licencia y no forman parte del repositorio.

## Cómo citar

Ver `CITATION.cff`.

## Agradecimientos

Esta investigación fue parcialmente apoyada por el HPC OCEANO (FONDEQUIP N° EQM170214).
