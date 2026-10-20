"""Rebuild draft figures from saved summaries; no training or result mutation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / 'experiments' / 'results'
OUT = PAPER / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.labelsize': 8, 'legend.fontsize': 7})


def read(relative):
    return json.loads((RESULTS / relative).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight', dpi=220)
    plt.close(fig)


def empirical():
    data = read('empirical/full_sgd_main/summary.json')
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout='constrained')
    for mode, color, label in [('square', '#176b9c', 'Square'), ('linear', '#bd5d25', 'Linear')]:
        rows = sorted([r for r in data['endpoints'] if r['energy_section'] == 1.5 and r['mode'] == mode], key=lambda r:r['ratio'])
        assert len(rows) == 3 and all(not r['missing_seeds'] for r in rows)
        stats = [r['encoder_displacement'] for r in rows]
        means = np.array([r['mean'] for r in stats])
        ci = np.array([r['seed_bootstrap_mean_ci95'] for r in stats])
        ax.errorbar([r['ratio'] for r in rows], means, yerr=np.array([means-ci[:,0], ci[:,1]-means]),
                    marker='o', color=color, label=label, capsize=3, lw=1.4, ms=4)
    ax.set(xscale='log', xticks=[.1,1,10], xticklabels=['0.1','1','10'],
           xlabel='Layerwise ratio r', ylabel='Encoder projector displacement', ylim=(0,None))
    ax.legend(frameon=False)
    save(fig, 'empirical_subspaces')


def interventions():
    data = read('empirical/full_sgd_interventions/summary.json')
    rows = sorted([r for r in data['seed_summaries'] if r['mode']=='square' and r['horizon']==1], key=lambda r:r['step'])
    assert [r['step'] for r in rows] == [500,2000,3500]
    fig, ax = plt.subplots(figsize=(3.35,2.5), layout='constrained')
    for key, color, label, marker in [
        ('sign_probability','#176b9c','Single stochastic update','o'),
        ('sign_match_of_replicate_mean','#347b51','Mean of 32 streams','s'),
        ('constant_decrease_probability','#929292','Always decrease','^')]:
        values = [np.array([s[key] for s in r['per_seed']])*100 for r in rows]
        means = [v.mean() for v in values]
        ax.plot([r['step'] for r in rows], means, marker=marker, color=color, label=label, lw=1.3, ms=4)
        if key == 'sign_probability':
            for r, v in zip(rows, values):
                ax.scatter(r['step']+np.linspace(-75,75,len(v)),v,s=7,color=color,alpha=.35,edgecolors='none')
    ax.set(xticks=[500,2000,3500],xlabel='Empirical training update',ylabel='Direction agreement (%)',ylim=(30,103))
    ax.legend(frameon=False,loc='lower left')
    save(fig, 'intervention_boundary')


def frozen():
    data = read('cifar/frozen_encoder_heads/summey.json')
    methods = ['encoder_full_ols','source_projection_ols','rank8_ls','trained_projection_ols','trained_classifier']
    labels = ['Full encoder OLS','Original head OLS','Rank-8 LS','CE head OLS','CE classifier']
    fig, ax = plt.subplots(figsize=(3.35,2.55), layout='constrained')
    for step, offset, color, label in [(0,-.13,'#929292','Initial encoder'),(4000,.13,'#176b9c','Pretrained encoder')]:
        for i, method in enumerate(methods):
            row=next(r for r in data['rows'] if r['source_step']==step and r['method']==method)
            vals=np.array(row['validation_accuracy']['values'])*100
            assert len(vals)==3
            ax.scatter(vals, i+offset+np.linspace(-.035,.035,3),s=12,alpha=.45,color=color,edgecolors='none')
            ax.scatter(vals.mean(), i+offset,s=25,marker='D',color=color,label=label if i==0 else None)
    ax.set(yticks=range(len(methods)),yticklabels=labels,xlabel='Validation accuracy (%)',xlim=(20,42))
    ax.invert_yaxis()
    ax.legend(frameon=False,loc='lower right',fontsize=6.5)
    save(fig, 'frozen_heads')


if __name__ == '__main__':
    empirical()
    interventions()
    frozen()
    print('Built three figures from saved JSON summaries; no training performed.')
