# TempBalance-LM

## Language Modeling Experiments of paper: Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training [NeurIPS 2023 Spotlight]

[Yefan Zhou](https://yefanzhou.github.io/), Tianyu Pang, [Keqin Liu](https://math.nju.edu.cn/szdw/apypl1/20221121/i231998.html), Charles H. Martin, [Michael W. Mahoney](https://www.stat.berkeley.edu/~mmahoney/), [Yaoqing Yang](https://sites.google.com/site/yangyaoqingcmu/)

[Full paper](https://openreview.net/forum?id=oyV9FslE3j)

## Install
```bash
bash install.sh
conda activate ww_train_lm
bash penn_tree.sh
```

## Experiments
```bash
# Baseline 
bash ./BTD-Transformer/scripts/tensorized/run_ptb.sh

# TempBalance
bash ./BTD-Transformer/scripts/tensorized/run_ptb_tb.sh
```