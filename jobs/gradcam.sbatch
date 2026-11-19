#!/bin/bash
# Grad-CAM de la semilla declarada (42) para M1 y M4, con la seleccion de imagenes fijada en
# configs/pilot_m1_m4.yaml (evaluation.gradcam). Lee las evaluaciones sobre test ya hechas por
# jobs/evaluar_final.sbatch; no entrena ni vuelve a evaluar, y no decide nada con el test.
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   GIT_COMMIT=<hash> sbatch --export=ALL,GPU_ID=1 jobs/gradcam.sbatch
#SBATCH -J gradcam
#SBATCH -p GPU
#SBATCH --qos=short-gpu
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH -o logs/%j_%x.out
#SBATCH -e logs/%j_%x.err

set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/work/$USER/isic}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis

# Sin esto TensorFlow no encuentra las librerias CUDA instaladas por pip.
NV_DIR=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV_DIR"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}

# Slurm no administra las GPU (Gres=null): las dos son visibles, se fija una por trabajo.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# El clon del clúster no trae .git: definir el commit del codigo copiado, si se conoce.
export GIT_COMMIT="${GIT_COMMIT:-no definido}"

cd "$HOME/proyecto_de_titulo"

echo "nodo: $(hostname)"
echo "fecha: $(date -Is)"
echo "datos: $DATA_ROOT | commit: $GIT_COMMIT"
nvidia-smi

for MODELO in m1 m4; do
    EVAL=$(ls -d runs/evaluacion/${MODELO}_seed42_test_* | sort | tail -1)
    echo "=== $MODELO: $EVAL ==="
    python -u scripts/generar_gradcam.py --evaluacion "$EVAL" --data-root "$DATA_ROOT" --confirmo-config-congelada
done
