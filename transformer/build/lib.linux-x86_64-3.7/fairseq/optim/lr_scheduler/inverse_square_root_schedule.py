# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import math
import numpy as np
import torch.nn as nn
from . import FairseqLRScheduler, register_lr_scheduler
from operator import itemgetter
import pandas as pd

def get_layer_temps(args, temp_balance, n_alphas, epoch_val):
    """naming of the variable is consistent with tianyu's code

    Args:
        temp_balance (_type_): method type 
        n_alphas (_type_): all the metric values
        epoch_val (_type_): basic untuned learning rate
    """
    n = len(n_alphas)
    idx = [i for i in range(n)]
    temps = np.array([epoch_val] * n)

    if temp_balance == 'tbr':
        print("--------------------> Use tbr method to schedule")
        idx = np.argsort(n_alphas)
        #temps = [2 * epoch_val * (0.35 + 0.15 * 2 * i / n) for i in range(n)]
        temps = [epoch_val * (args.lr_min_ratio + args.lr_slope * i / n) for i in range(n)]
        #print("temps",    args.lr_min_ratio,  args.lr_slope )
        #print("temps", temps)
        # Examples:
        # 4 3 5 -> argsort -> 1 0 2
        # temps = [0.7, 1, 1.3]
        # zip([1, 0, 2], [0.7, 1, 1.3]) -> [(1, 0.7), (0, 1), (2, 1.3)] -> [(0, 1),(1, 0.7),(2, 1.3)]
        return [value for _, value in sorted(list(zip(idx, temps)), key=itemgetter(0))]
    elif temp_balance == 'tb_linear_map':
        #print("!!!!!!!!!!", epoch_val, args.lr_min_ratio, args.lr_min_ratio + args.lr_slope)
        lr_range = [args.lr_min_ratio * epoch_val,  (args.lr_min_ratio + args.lr_slope) * epoch_val]
        score_range = [min(n_alphas),  max(n_alphas)]
        temps = np.interp(n_alphas, score_range, lr_range)
        #print(temps)
        return temps
    
    elif temp_balance == 'tb_sqrt':
        temps = np.sqrt(n_alphas)/np.sum(np.sqrt(n_alphas)) * n * epoch_val
        return temps
    
    elif temp_balance == 'tb_log2':
        temps = np.log2(n_alphas)/np.sum(np.log2(n_alphas)) * n * epoch_val
        return temps
    else:
        raise NotImplementedError


def net_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False,
            init_stable_rank=None,
            sr_mid_pos=None):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'stable_rank':[],
        'norm_stable_rank':[],
        'init_norm_stable_rank':[],
        'eig_ratio':[],
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone()
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()

            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                sr_eigs = eigs

            if sr_mid_pos is not None:
                mid = int(len(sr_eigs) / sr_mid_pos)
                sr_eigs = sr_eigs[mid: ]
            eigs_sum = torch.sum(sr_eigs)
            max_eigs = torch.max(sr_eigs)
            stable_rank = eigs_sum / max_eigs
            norm_stable_rank = eigs_sum / len(sr_eigs)
            mid_eig = sr_eigs[len(sr_eigs) // 2]
            eig_ratio = max_eigs / mid_eig

            results['stable_rank'].append(stable_rank.item())
            results['norm_stable_rank'].append(norm_stable_rank.item())
            if init_stable_rank is not None:
                results['init_norm_stable_rank'].append(norm_stable_rank.item() / init_stable_rank[len(results['init_norm_stable_rank'])])
            else:
                results['init_norm_stable_rank'].append(0)
            results['eig_ratio'].append(eig_ratio.item())
            
            if filter_zeros:
                #print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    #print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                #print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()

            results['spectral_norm'].append(spectral_norm)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())
    
    return results

def net_cul_esd_estimator(
            net=None,
            ori_net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False,
            init_stable_rank=None,
            sr_mid_pos=None):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'stable_rank':[],
        'norm_stable_rank':[],
        'init_norm_stable_rank':[],
        'eig_ratio':[],
        }
    print("=================================")
    print(f"anslyzing the cumulated weight matrices: fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for (name, m), (ori_name, ori_m) in zip(net.named_modules(), ori_net.named_modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone()
            ori_matrix = ori_m.weight.data.clone()
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
                ori_matrix = torch.flatten(ori_matrix, start_dim=2) * math.sqrt(conv_norm)
                ori_matrix = ori_matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            eigs = torch.square(torch.linalg.svdvals(matrix - ori_matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()


            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                sr_eigs = eigs

            if sr_mid_pos is not None:
                mid = int(len(sr_eigs) / sr_mid_pos)
                sr_eigs = sr_eigs[mid: ]

            eigs_sum = torch.sum(sr_eigs)
            max_eigs = torch.max(sr_eigs)
            stable_rank = eigs_sum / max_eigs
            norm_stable_rank = eigs_sum / len(sr_eigs)
            mid_eig = sr_eigs[len(sr_eigs) // 2]
            eig_ratio = max_eigs / mid_eig

            results['stable_rank'].append(stable_rank.item())
            results['norm_stable_rank'].append(norm_stable_rank.item())
            if init_stable_rank is not None:
                results['init_norm_stable_rank'].append(norm_stable_rank.item() / init_stable_rank[len(results['init_norm_stable_rank'])])
            else:
                results['init_norm_stable_rank'].append(0)
            results['eig_ratio'].append(eig_ratio.item())
            
            if filter_zeros:
                #print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    #print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                #print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()

            results['spectral_norm'].append(spectral_norm)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())
    
    return results

@register_lr_scheduler('inverse_sqrt')
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer, model, ori_model=None):
        super().__init__(args, optimizer, model)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        print("--------------------> Initialize the InverseSquareRootSchedule")
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.model = model
        self.optimizer.set_lr(self.lr)
        self.metrics_score = None
        self.layer_stats = None

        self.ori_model = ori_model

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""

        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5

        #print(f"################{self.lr}")
        if num_updates % self.args.esd_interval == 0:
            print(f"--------------------> Update the metrics values")
            metrics = net_esd_estimator(self.model, 
                    EVALS_THRESH = 0.00001,
                    bins = 100,
                    fix_fingers=self.args.fix_fingers,
                    xmin_pos=self.args.xmin_pos,
                    filter_zeros=self.args.filter_zeros=='True')
            
            self.layer_stats= pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
            self.metrics_score = np.array(self.layer_stats[self.args.metric])
            #print(self.metrics_score)
            np.save(os.path.join(self.args.save_dir, f'metrics_update{num_updates}.npy'), metrics)

            if self.ori_model is not None:
                print(f"--------------------> Compute cumulative weight changes")
                cul_metrics = net_cul_esd_estimator(self.model, 
                                self.ori_model,
                                EVALS_THRESH = 0.00001,
                                bins = 100,
                                fix_fingers=self.args.fix_fingers,
                                xmin_pos=self.args.xmin_pos,
                                filter_zeros=self.args.filter_zeros=='True')
                np.save(os.path.join(self.args.save_dir, f'cul_metrics_update{num_updates}.npy'), cul_metrics)

        if self.args.tb == 'True' \
                    and (self.args.tbr_after_warm == 'False' \
                                or (num_updates >= self.args.warmup_updates)):

            scheduled_lr = get_layer_temps(self.args, 
                                            self.args.temp_balance_lr, 
                                            self.metrics_score, self.lr)
            
            self.layer_stats['scheduled_lr'] = scheduled_lr
            layer_name_to_tune = list(self.layer_stats['longname'])
            all_params_lr = []
            c = 0
            linear_count, norm1_count, norm2_count, norm3_count = 0,0,0,0
            for name, _ in self.model.named_modules():
                if name in layer_name_to_tune:
                    scheduled_lr = self.layer_stats[self.layer_stats['longname'] == name]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    linear_count += 1
                    c = c + 1
                elif self.args.batchnorm == 'True' \
                        and 'self_attn_layer_norm' in name \
                                and name.replace('self_attn_layer_norm', 'self_attn.out_proj') in layer_name_to_tune:
                    scheduled_lr = \
                        self.layer_stats[self.layer_stats['longname'] == name.replace('self_attn_layer_norm', 'self_attn.out_proj')]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    norm1_count += 1
                    c = c + 1
                elif self.args.batchnorm == 'True' \
                        and 'final_layer_norm' in name \
                                and name.replace('final_layer_norm', 'fc2') in layer_name_to_tune:
                    scheduled_lr = \
                        self.layer_stats[self.layer_stats['longname'] == name.replace('final_layer_norm', 'fc2')]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    norm2_count += 1
                    c = c + 1
                elif self.args.batchnorm == 'True' \
                        and 'encoder_attn_layer_norm' in name \
                                and name.replace('encoder_attn_layer_norm', 'encoder_attn.out_proj') in layer_name_to_tune:
                    scheduled_lr = \
                        self.layer_stats[self.layer_stats['longname'] == name.replace('encoder_attn_layer_norm', 'encoder_attn.out_proj')]['scheduled_lr'].item()
                    all_params_lr.append(scheduled_lr)
                    norm3_count += 1
                    c = c + 1

            for index, param_group in enumerate(self.optimizer._optimizer.param_groups):
                if index <= c - 1:
                    param_group['lr'] = all_params_lr[index]
                else:
                    param_group['lr'] = self.lr

            if num_updates % self.args.esd_interval == 0:
                # save layer stats
                print(f"--------------------> Save the layer stats")
                self.layer_stats.to_csv(os.path.join(self.args.save_dir, f'layer_stats_update{num_updates}.csv'))

        elif self.args.acc_v == 'True' \
                and (self.args.tbr_after_warm == 'False' \
                    or (num_updates >= self.args.warmup_updates)):

            layer_name_to_tune = list(self.layer_stats['longname'])
            all_params_lr = []
            v_layer_count = 0
            v_layers = -1
            c = 0
            for name, _ in self.model.named_modules():
                if 'v_proj' in name:
                    v_layers += 1
            print(f'----------> number of V layers: {v_layers + 1}')
            esd_lr = []
            for name, _ in self.model.named_modules():
                if 'v_proj' in name and 'decoder' in name:
                    # the lr for v_layers are evenly distributed between base_lr * lr_min and base_lr * (lr_min + lr_slope)
                    v_lr = self.lr * (self.args.lr_min_ratio + self.args.lr_slope * (v_layers - v_layer_count) / v_layers)
                    all_params_lr.append(v_lr)
                    v_layer_count += 1
                    c = c + 1
                    if name in layer_name_to_tune:
                        esd_lr.append(v_lr)
                elif name in layer_name_to_tune:
                    all_params_lr.append(self.lr)
                    c = c + 1
                    if name in layer_name_to_tune:
                        esd_lr.append(self.lr)

            optim_layer_num = 0
            for index, param_group in enumerate(self.optimizer._optimizer.param_groups):
                if index <= c - 1:
                    param_group['lr'] = all_params_lr[index]
                else:
                    param_group['lr'] = self.lr

            # compare the layers in optimizer and layers in model
            if num_updates % self.args.esd_interval == 0:
                layer_stats_len = len(self.layer_stats['longname'])
                print(f"--------------------> Save the layer stats")
                print(f'# of layers in model: {c}, # of layers in optimizer: {optim_layer_num}, # of layers in esd analysis: {layer_stats_len}')
                self.layer_stats['scheduled_lr'] = esd_lr
                self.layer_stats.to_csv(os.path.join(self.args.save_dir, f'layer_stats_update{num_updates}.csv'))
                    
        else:
            self.optimizer.set_lr(self.lr)
        
        return self.lr
