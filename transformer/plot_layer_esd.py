import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import io
from contextlib import redirect_stdout, redirect_stderr
import powerlaw

from collections import defaultdict
import matplotlib as mpl
import numpy as np

params = {'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':12,
         'ytick.labelsize':15,
         'axes.titlepad': 1,
         'axes.labelpad': 1,
         'axes.grid': True,
          'legend.fontsize': 15
    }
mpl.rcParams.update(params)

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import matplotlib.cbook


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json

# c = 'steelblue'
# base_c = 'lightskyblue'
import seaborn as sns
c = 'coral'
c1 = 'yellowgreen'
base_c = 'aqua'

plt.rcParams['text.usetex'] = False
sns.set(style="ticks")
plt.rc('font', family='serif')
plt.rc('font', serif='STIXGeneral')

yaxisfont=45
bigfont = 27 + 5
medfont = 15 
smallfont = 17

plt.rcParams['xtick.labelsize']=medfont
plt.rcParams['ytick.labelsize']=medfont
plt.rcParams['axes.titlesize']=medfont

sns.set_style('ticks', {'font.family':'serif', 'font.serif':'Times New Roman', 'font.size': 16})

sns.set_palette("muted")
cmap = sns.color_palette("tab10")
ticks_fontsize = 18
label_fontsize = 20
cbar_fontsize = 18


EVALS_THRESH = 0.00001
thresh = EVALS_THRESH
XMIN_PEAK = 'xmin_peak'
XMIN_MID = 'xmin_mid'
POWER_LAW = 'power_law'
TRUNCATED_POWER_LAW='truncated_power_law'
xmin_pos = 2

def pl_fit(data=None, xmin=None, xmax=None, verbose=False, distribution='PL'):
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        return powerlaw.Fit(data, xmin=xmin, xmax=xmax, 
                verbose=verbose, distribution=distribution, 
                xmin_distribution=distribution)

def plot_loghist(x, bins, xmin, legend, color):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, density=True, alpha=0.2, label=legend, color=color)

    if xmin:
        plt.axvline(xmin, color=color, label=r'$\lambda_{min}$')

    plt.xscale('log')

task = 'mt_iwslt14_de_en'
model = 'transformer_lm'
config_dict = {
    'baseline': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/baseline/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43',
    'acc_mid_0.2_1.6': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/adam_acc_mid/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/min_0.2_slope1.6',
    'acc_mid_0.4_1.2': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/adam_acc_mid/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/min_0.4_slope1.2',
    'acc_mid_0.5_1.0': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/adam_acc_mid/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/min_0.5_slope1.0',
    'acc_mid_0.6_0.8': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/adam_acc_mid/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/min_0.6_slope0.8',
    'acc_mid_0.8_0.4': '/data/yefan0726/checkpoints/zihang/checkpoints/nlp/mt/iwslt14_de_en/adam_acc_mid/transformer_iwslt_de_en_v2_iwslt14_de_en_seed43/min_0.8_slope0.4',
}


distribution = 'power_law' #truncated_ truncated_
fit_type=distribution
fix_fingers="xmin_mid" #"xmin_peak"
c_idx = 0

for config, ckpt_dir in config_dict.items():
    seed_dir = ckpt_dir

    first_esd = np.load(os.path.join(seed_dir, 'metrics_update0.npy'), allow_pickle=True).item()
    for idx, layer_name in enumerate(first_esd['longname']):
        xmin_list = []
        alpha_list = []
        xmax_list = []
        for step in range(0, 60500, 500):
            esd = np.load(os.path.join(seed_dir, f'metrics_update{step}.npy'), allow_pickle=True).item()
            evals = esd['eigs'][idx]
            plt.figure(figsize=(10, 8))
            title = f'{layer_name}, fit: {fix_fingers} \n'
            xmax = np.max(evals)
            xmax_list.append(xmax)

            if fix_fingers==XMIN_PEAK:
                print("fix the fingers by setting xmin to the peak of the ESD")
                # nz_evals = evals[evals > thresh]
                nz_evals = evals
                num_bins = 100  # np.min([100, len(nz_evals)])
                h = np.histogram(np.log10(nz_evals), bins=num_bins)
                ih = np.argmax(h[0])
                xmin2 = 10 ** h[1][ih]
                xmin_range = (0.95 * xmin2, 1.05 * xmin2)
                fit = pl_fit(data=nz_evals, xmin=xmin_range,
                    xmax=xmax, verbose=False, 
                    distribution=distribution)

            elif fix_fingers==XMIN_MID:
                # print("fix the fingers by setting xmin to the mid of the ESD")
                # nz_evals = evals[evals > thresh]
                nz_evals = evals
                i = int(len(nz_evals) / xmin_pos)
                xmin = nz_evals[i]
                fit = pl_fit(data=nz_evals, xmin=xmin,
                    xmax=xmax, verbose=False, 
                    distribution=distribution)

            else: 
                print("powerlaw.Fit no xmin , distribution={} ".format(distribution))
                # nz_evals = evals[evals > thresh]
                nz_evals = evals
                fit = pl_fit(data=nz_evals, xmax=xmax, verbose=False, distribution=distribution) 
            
            xmin = fit.xmin
            alpha = fit.alpha
            D = fit.D
            if fit_type==TRUNCATED_POWER_LAW:
                alpha = fit.truncated_power_law.alpha
                Lambda = fit.truncated_power_law.Lambda
                D = fit.truncated_power_law.D

            xmin_list.append(xmin)
            alpha_list.append(alpha)
            fig2 = fit.plot_pdf(color='b', linewidth=0) # invisbile
            fig2 = fit.plot_pdf(color=cmap[c_idx], linewidth=2) #color='orange',
            if fit_type==POWER_LAW:
                fit.power_law.plot_pdf(color=cmap[c_idx], linestyle='--', ax=fig2) #color='r',
                title = title + rf"Epoch: best $\alpha=${alpha:.3f}; " + \
                    r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
                    r"$\lambda_{min}=$"+"{0:.3f}".format(xmin) + \
                    f'num of eigs: {len(nz_evals)}' "\n" 
            else:
                fit.truncated_power_law.plot_pdf(color=cmap[c_idx], linestyle='--', ax=fig2)
                title = title + rf"Epoch: best $\alpha=${alpha:.3f}; " + \
                    r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
                    r'$Lambda=$'+"{0:.3f};".format(Lambda) + \
                    r"$\lambda_{min}=$"+"{0:.3f}".format(xmin) + "\n"

            plot_loghist(evals[evals>(xmin/100)], 
                        bins=100, xmin=xmin, legend=f'Epoch=best', color=cmap[c_idx])
            
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            # make new dir if not exist
            if not os.path.exists(f"/data/yefan0726/checkpoints/zihang/figures/esd/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}"):
                os.makedirs(f"/data/yefan0726/checkpoints/zihang/figures/esd/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}")

            plt.savefig(f"/data/yefan0726/checkpoints/zihang/figures/esd/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}/esd_step_{step}.png")
            plt.close()


        # plot xmin and alpha:
        fig, axs = plt.subplots(1, 3, figsize=(30, 5))
        axs[0].plot(xmin_list)
        axs[1].plot(xmax_list)
        axs[2].plot(alpha_list)

        axs[0].set_xlabel('step', fontsize=20)
        axs[0].set_ylabel('eigenvalue', fontsize=20)
        axs[0].set_title(r"$\lambda_{min}$" + f" for {layer_name}", fontsize=20)
        # axs[0].legend()

        axs[1].set_xlabel('step', fontsize=20)
        axs[1].set_ylabel('eigenvalue', fontsize=20)
        axs[1].set_title(r"$\lambda_{max}$" + f" for {layer_name}", fontsize=20)
        # axs[0].legend()

        axs[2].set_xlabel('step', fontsize=20)
        axs[2].set_ylabel('Alpha', fontsize=20)
        axs[2].set_title(r"$\alpha$" f' for {layer_name}', fontsize=20)
        
        if not os.path.exists(f"/data/yefan0726/checkpoints/zihang/figures/training_dynamic/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}"):
            os.makedirs(f"/data/yefan0726/checkpoints/zihang/figures/training_dynamic/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}")

        plt.savefig(f"/data/yefan0726/checkpoints/zihang/figures/training_dynamic/{task}/{model}/{config}/fix_{fix_fingers}/layer_{layer_name}/metrics.png")
        plt.close()