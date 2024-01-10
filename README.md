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

## Usage
```python
from tempbalance import Tempbalance
import torch
model = ...
# initialize necessary hyperparameters
start_lr = ...
total_steps = ...
# initialize the scheduler
tb_scheduler = Tempbalance(net=model,
                start_lr=start_lr,
                total_steps=total_steps,
                lr_min_ratio=0.5,
                lr_max_ratio=1.5
                )
# initialize optimizer parameter group
tb_param_group = tb_scheduler.build_optimizer_param_group(untuned_lr=0.1)
optimizer = optim.SGD(
    tb_param_group,
    ...
)
# training loop
for epoch in range(1, ...):
    ...
    train()
    test()
    # get global decayed learning rate
    untuned_global_lr = some_torch_lr_scheduler(epoch)
    # temperature balancing
    tb_scheduler.step(optimizer, untuned_global_lr)
    ...
```


## Experiments
```bash
# Baseline 
bash ./BTD-Transformer/scripts/tensorized/run_ptb.sh

# TempBalance
bash ./BTD-Transformer/scripts/tensorized/run_ptb_tb.sh
```

## Acknowledgement
1. We thank the open-sourced codebase [The-compression-of-Transformer](https://github.com/szhangtju/The-compression-of-Transformer/tree/master).
