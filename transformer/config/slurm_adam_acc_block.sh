#!/bin/bash
#SBATCH -p rise
#SBATCH --array=7-11
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=steropes
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

export PYTHONUNBUFFERED=1
bash ~/.bashrc
conda activate ww_train

SEED=43
DATA_PATH=/data/yefan0726/data/nlp/mt/data-bin/iwslt14.tokenized.de-en.joined
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
acc_block=True
tbr_after_warm=True
temp_balance_lr=tb_linear_map
metric=alpha

xmin_pos=2
fix_fingers=xmin_mid

cd /home/eecs/yefan0726/zihang/layer-wise-learning-rate-schedule-/transformer

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p /home/eecs/yefan0726/zihang/layer-wise-learning-rate-schedule-/transformer/config/config.txt)

lr_scale=$(echo $cfg | cut -f 1 -d ' ')

OUTPUT_PATH=/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/${PROBLEM}/adam_acc_block/${ARCH}_${PROBLEM}_seed${SEED}/lr_scale_${lr_scale}
NUM=5
mkdir -p $OUTPUT_PATH

# python train.py ${DATA_PATH} \
#         --seed ${SEED} \
#         --adam-eps 1e-08 \
#         --arch ${ARCH} --share-all-embeddings \
#         --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#         --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
#         --criterion label_smoothed_cross_entropy \
#         --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
#         --lr 0.0015 --min-lr 1e-9 \
#         --label-smoothing 0.1 --weight-decay 0.0001 \
#         --tb False --acc-v False --acc-mid False \
#         --acc-block $acc_block --lr-scale $lr_scale --tbr-after-warm ${tbr_after_warm} --temp-balance-lr ${temp_balance_lr} --metric ${metric} \
#         --xmin-pos ${xmin_pos} --fix-fingers ${fix_fingers} \
#         --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
#         --update-freq 1 --no-progress-bar --log-interval 50 \
#         --ddp-backend no_c10d \
#         --keep-last-epochs ${NUM} --max-epoch 55 \
#         --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
#         | tee -a ${OUTPUT_PATH}/train_log.txt

# # --early-stop ${NUM} \

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
> ${RESULT_PATH}/res.txt

# scp -r ${OUTPUT_PATH}/train_log.txt yefan0726@watson.millennium.berkeley.edu:/home/eecs/yefan0726/zihang/layer-wise-learning-rate-schedule-/transformer/results/train_log_${lr_scale}.txt
