#!/bin/bash
#SBATCH --array=1-5
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=ace
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

pwd
hostname
date
echo starting job...
source ~/.bashrc

conda activate ww_train_2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

SEED=43
DATA_PATH=/scratch/tpang/zhliu/data/nlp/mt/data-bin/iwslt14.tokenized.de-en.joined
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
OUTPUT_PATH=/scratch/tpang/zhliu/checkpoints/nlp/mt/${PROBLEM}/baseline/${ARCH}_${PROBLEM}_seed${SEED}
NUM=5

cd /scratch/tpang/zhliu/repos/layer-wise-learning-rate-schedule-/transformer


BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
    --quiet \
> ${RESULT_PATH}/avg_res.txt

CKPT=checkpoint_best.pt
python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
    --quiet \
> ${RESULT_PATH}/best_res.txt
