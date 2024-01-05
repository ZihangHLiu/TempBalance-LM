#!/bin/bash
#SBATCH --array=53-54
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=pavia
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
tb=True
tbr_after_warm=True
temp_balance_lr=tb_linear_map
metric=alpha
lr_min_ratio=$1
lr_slope=$2
xmin_pos=2
fix_fingers=xmin_mid

OUTPUT_PATH=/scratch/tpang/zhliu/checkpoints/nlp/mt/${PROBLEM}/adam_tb/${ARCH}_${PROBLEM}_seed${SEED}/after_warmup_${tbr_after_warm}_${temp_balance_lr}_${metric}/min_${lr_min_ratio}_slope${lr_slope}_${fix_fingers}_mid_pos${xmin_pos}
NUM=5
mkdir -p $OUTPUT_PATH

cd /scratch/tpang/zhliu/repos/layer-wise-learning-rate-schedule-/transformer

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

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
