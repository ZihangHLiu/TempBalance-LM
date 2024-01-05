export PYTHONUNBUFFERED=1
SEED=43
DATA_PATH=/scratch/tpang/zhliu/data/nlp/mt/data-bin/iwslt14.tokenized.de-en.joined
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
OUTPUT_PATH=/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/${PROBLEM}/baseline/${ARCH}_${PROBLEM}_seed${SEED}
NUM=5
mkdir -p $OUTPUT_PATH

python train.py ${DATA_PATH} \
    --seed ${SEED} \
    --adam-eps 1e-08 \
    --arch ${ARCH} --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --criterion label_smoothed_cross_entropy \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
    --lr 0.0015 --min-lr 1e-9 \
    --tb False \
    --label-smoothing 0.1 --weight-decay 0.0001 \
    --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
    --update-freq 1 --no-progress-bar --log-interval 50 \
    --ddp-backend no_c10d \
    --keep-last-epochs ${NUM} --max-epoch 55 \
    --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
    | tee -a ${OUTPUT_PATH}/train_log.txt

# --early-stop ${NUM} \

# python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

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

# fairseq-generate /scratch/tpang/zhliu/data/nlp/mt/data-bin/iwslt14.tokenized.de-en.joined \
#     --path /scratch/tpang/zhliu/checkpoints/nlp/mt/iwslt14_de_en/baseline/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe
# Generate test with beam=5: BLEU4 = 35.28, 68.3/43.0/28.9/19.9 (BP=0.978, ratio=0.978, syslen=128334, reflen=131156)