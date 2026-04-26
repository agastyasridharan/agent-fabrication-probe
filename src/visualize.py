"""
Generate publication-quality figures for the confabulation probe project.

Outputs to figures/ directory as both PNG (300dpi) and PDF.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)

# Style
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.8,
})

# Colors
FABRICATE_COLOR = '#E63946'
ADMIT_COLOR = '#457B9D'
ACCENT = '#2A9D8F'
DARK = '#1D3557'
LIGHT_GRAY = '#F1FAEE'
WARN_COLOR = '#E9C46A'
GRID_COLOR = '#E0E0E0'

PROMPT_COLORS = {
    'honesty': '#457B9D',
    'pressure': '#E9C46A',
    'neutral': '#A8DADC',
    'balanced': '#2A9D8F',
    'expert': '#E63946',
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with open(ROOT / 'data/trajectories/all_trajectories.json') as f:
    ALL_TRAJS = json.load(f)
with open(ROOT / 'data/labels.json') as f:
    LABEL_DATA = json.load(f)
with open(ROOT / 'data/intervention_results.json') as f:
    INTV = json.load(f)
with open(ROOT / 'data/transfer_results.json') as f:
    TRANSFER = json.load(f)

JUDGE = LABEL_DATA['judge_labels']
CAL_IDS = set(LABEL_DATA['calibration_ids'])


def heuristic_label(resp):
    n = re.compile(r'\b\d+\.?\d*\s*(?:eV|g/cm|meV|J|kJ|GPa)')
    e = re.compile(r'(?:approximately|around|roughly|about|estimated?|~)\s*\d+\.?\d*', re.I)
    r = re.compile(r'\d+\.?\d*\s*[-\u2013to]+\s*\d+\.?\d*\s*(?:eV|g/cm|meV)', re.I)
    return 1 if (n.search(resp) or e.search(resp) or r.search(resp)) else 0


def load_act(tid, layer):
    return np.load(ROOT / f'data/trajectories/activations/{tid}.npy')[layer]


EMPTY = [t for t in ALL_TRAJS if t['side'] == 'empty']
BAL = [t for t in EMPTY if t['system_prompt_variant'] == 'balanced']
for t in BAL:
    t['label'] = heuristic_label(t['assistant_response'])

# Material split (same seed as probe.py)
FORMULAS = sorted(set(t['formula'] for t in BAL))
random.shuffle(FORMULAS)
N_TRAIN = int(0.7 * len(FORMULAS))
TRAIN_F = set(FORMULAS[:N_TRAIN])
TEST_F = set(FORMULAS[N_TRAIN:])
TRAIN = [t for t in BAL if t['formula'] in TRAIN_F]
TEST = [t for t in BAL if t['formula'] in TEST_F]


def savefig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png', facecolor='white')
    fig.savefig(FIG_DIR / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    print(f'  Saved {name}')


# ---------------------------------------------------------------------------
# Figure 1: Layer Sweep
# ---------------------------------------------------------------------------

def fig_layer_sweep():
    print('Figure 1: Layer sweep')
    train_aurocs, test_aurocs = [], []
    for layer in range(33):
        tr_X = np.stack([load_act(t['trajectory_id'], layer) for t in TRAIN])
        tr_y = np.array([t['label'] for t in TRAIN])
        te_X = np.stack([load_act(t['trajectory_id'], layer) for t in TEST])
        te_y = np.array([t['label'] for t in TEST])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(tr_X, tr_y)
        train_aurocs.append(roc_auc_score(tr_y, clf.predict_proba(tr_X)[:, 1]))
        test_aurocs.append(roc_auc_score(te_y, clf.predict_proba(te_X)[:, 1]))

    best_layer = np.argmax(test_aurocs)
    layers = np.arange(33)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Shaded region between train and test
    ax.fill_between(layers, test_aurocs, train_aurocs, alpha=0.08, color=DARK)

    ax.plot(layers, train_aurocs, 'o-', color=DARK, markersize=4, alpha=0.5,
            label='Train AUROC', zorder=3)
    ax.plot(layers, test_aurocs, 's-', color=FABRICATE_COLOR, markersize=5,
            label='Test AUROC', zorder=4)

    # Best layer marker
    ax.annotate(f'Peak: layer {best_layer}\nAUROC = {test_aurocs[best_layer]:.3f}',
                xy=(best_layer, test_aurocs[best_layer]),
                xytext=(best_layer - 8, test_aurocs[best_layer] + 0.07),
                fontsize=10, fontweight='bold', color=FABRICATE_COLOR,
                arrowprops=dict(arrowstyle='->', color=FABRICATE_COLOR, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=FABRICATE_COLOR, alpha=0.9))

    # Success threshold
    ax.axhline(0.75, color=ACCENT, linestyle=':', alpha=0.7, linewidth=1.2)
    ax.text(26, 0.72, 'Success threshold (0.75)', fontsize=8, color=ACCENT,
            ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # Chance line
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(26, 0.505, 'Chance', fontsize=8, color='gray', alpha=0.5)

    # Layer regions
    for x0, x1, label in [(0, 8, 'Early layers'), (8, 24, 'Mid layers'), (24, 32, 'Late layers')]:
        ax.axvspan(x0, x1, alpha=0.03, color=DARK)
        ax.text((x0 + x1) / 2, 0.46, label, fontsize=8, color='gray',
                ha='center', alpha=0.6)

    ax.set_xlabel('Layer')
    ax.set_ylabel('AUROC')
    ax.set_title('Fabrication Probe: Per-Layer AUROC on Material-Split Test Set')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(-0.5, 32.5)
    ax.set_ylim(0.43, 1.04)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)

    savefig(fig, '01_layer_sweep')
    return test_aurocs, best_layer


# ---------------------------------------------------------------------------
# Figure 2: System Prompt Fabrication Rates
# ---------------------------------------------------------------------------

def fig_prompt_rates():
    print('Figure 2: System prompt fabrication rates')
    prompts = ['honesty', 'pressure', 'neutral', 'balanced', 'expert']
    rates = []
    for sp in prompts:
        sub = [t for t in EMPTY if t['system_prompt_variant'] == sp]
        fab = sum(heuristic_label(t['assistant_response']) for t in sub)
        rates.append(fab / len(sub))

    fig, ax = plt.subplots(figsize=(10, 4.5))

    prompt_labels = ['honesty', 'pressure', 'neutral', 'balanced', 'expert']
    bars = ax.barh(prompt_labels, rates, color=[PROMPT_COLORS[p] for p in prompts],
                   edgecolor='white', linewidth=1.5, height=0.55)

    # Labels: always to the right of the bar, never inside
    for bar, rate, prompt in zip(bars, rates, prompts):
        x_pos = max(rate, 0.01) + 0.015
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1%}', ha='left', va='center', fontweight='bold',
                color=PROMPT_COLORS[prompt], fontsize=13)

    ax.set_xlabel('Fabrication Rate')
    ax.set_title('Fabrication Rate by System Prompt')
    ax.set_xlim(0, 1.25)
    ax.invert_yaxis()

    # Annotations: placed well clear of percentage labels
    ax.annotate('"Approximate estimates are acceptable"',
                xy=(0.18, 1), fontsize=8.5, color='gray',
                fontstyle='italic', va='center')
    ax.annotate('"Draw on your deep knowledge"',
                xy=(0.18, 4), fontsize=8.5, color='gray',
                fontstyle='italic', va='center')

    ax.grid(True, axis='x', alpha=0.15, color=GRID_COLOR)
    savefig(fig, '02_prompt_fabrication_rates')


# ---------------------------------------------------------------------------
# Figure 3: ROC Curve
# ---------------------------------------------------------------------------

def fig_roc(best_layer):
    print('Figure 3: ROC curve')
    tr_X = np.stack([load_act(t['trajectory_id'], best_layer) for t in TRAIN])
    tr_y = np.array([t['label'] for t in TRAIN])
    te_X = np.stack([load_act(t['trajectory_id'], best_layer) for t in TEST])
    te_y = np.array([t['label'] for t in TEST])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    clf.fit(tr_X, tr_y)
    proba = clf.predict_proba(te_X)[:, 1]
    fpr, tpr, _ = roc_curve(te_y, proba)
    auroc = roc_auc_score(te_y, proba)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Shaded area under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=FABRICATE_COLOR)
    ax.plot(fpr, tpr, color=FABRICATE_COLOR, linewidth=2.5,
            label=f'Probe (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1, label='Random (0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve at Layer {best_layer}')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)

    savefig(fig, '03_roc_curve')
    return clf


# ---------------------------------------------------------------------------
# Figure 4: Intervention Comparison
# ---------------------------------------------------------------------------

def fig_intervention():
    print('Figure 4: Intervention comparison')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.4]})

    # Left panel: bar chart of baseline vs injection
    methods = ['Baseline\n(no intervention)', 'Prompt\ninjection']
    rates = [INTV['baseline_fab_rate'], INTV['injection_fab_rate']]
    colors = ['#A8DADC', ACCENT]

    bars = ax1.bar(methods, rates, color=colors, edgecolor='white',
                   linewidth=1.5, width=0.5)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold',
                fontsize=14, color=DARK)

    # Arrow showing reduction
    ax1.annotate('', xy=(1, rates[1] + 0.04), xytext=(0, rates[0] + 0.04),
                arrowprops=dict(arrowstyle='->', color=FABRICATE_COLOR, lw=2))
    ax1.text(0.5, (rates[0] + rates[1]) / 2 + 0.06,
            f'{INTV["injection_reduction"]:.0%}\nreduction',
            ha='center', va='center', fontweight='bold', fontsize=11,
            color=FABRICATE_COLOR)

    ax1.set_ylabel('Fabrication Rate')
    ax1.set_title('Prompt Injection Intervention')
    ax1.set_ylim(0, 0.85)
    ax1.grid(True, axis='y', alpha=0.15, color=GRID_COLOR)

    # Right panel: steering alpha sweep
    alphas = sorted(INTV['steering'].keys(), key=float)
    steer_rates = [INTV['steering'][a]['fab_rate'] for a in alphas]
    alpha_vals = [float(a) for a in alphas]

    ax2.axhline(INTV['baseline_fab_rate'], color='gray', linestyle='--',
               alpha=0.5, linewidth=1.2, label='Baseline')
    ax2.axhline(INTV['injection_fab_rate'], color=ACCENT, linestyle='-',
               alpha=0.7, linewidth=1.5, label='Prompt injection')

    ax2.plot(alpha_vals, steer_rates, 'o-', color=FABRICATE_COLOR,
            markersize=6, label='Activation steering', zorder=3)

    # Shade the "makes it worse" region
    worse_mask = [r > INTV['baseline_fab_rate'] for r in steer_rates]
    for i in range(len(alpha_vals)):
        if worse_mask[i]:
            ax2.plot(alpha_vals[i], steer_rates[i], 'o', color=FABRICATE_COLOR,
                    markersize=8, zorder=4)
            ax2.annotate('worse', xy=(alpha_vals[i], steer_rates[i]),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=7, color='gray', ha='center')

    ax2.set_xlabel('Steering Strength (\u03b1)')
    ax2.set_ylabel('Fabrication Rate')
    ax2.set_title('Activation Steering Sweep')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 0.85)
    ax2.grid(True, alpha=0.15, color=GRID_COLOR)

    fig.suptitle('Intervention Results', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, '04_intervention')


# ---------------------------------------------------------------------------
# Figure 5: Generalization / Transfer
# ---------------------------------------------------------------------------

def fig_transfer():
    print('Figure 5: Generalization')
    fig, ax = plt.subplots(figsize=(9, 5))

    categories = [
        'Within-distribution\n(material-split)',
        'Cross-chemistry\n(oxides -> sulfides)',
        'Cross-tool\n(MP -> ChemDB)',
        'Cross-template\n(novel paraphrases)',
    ]
    aurocs = [0.795, 0.787, TRANSFER['cross_tool_auroc'], TRANSFER['cross_template_auroc']]
    colors = [DARK, ADMIT_COLOR, FABRICATE_COLOR, ACCENT]

    bars = ax.bar(categories, aurocs, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.55)

    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, auroc + 0.01,
               f'{auroc:.3f}', ha='center', va='bottom', fontweight='bold',
               fontsize=13, color=DARK)

    # Thresholds
    ax.axhline(0.75, color=ACCENT, linestyle=':', alpha=0.6, linewidth=1.2)
    ax.text(3.7, 0.755, 'Probe\nthreshold\n(0.75)', fontsize=8, color=ACCENT,
           ha='right', va='bottom')
    ax.axhline(0.65, color=WARN_COLOR, linestyle=':', alpha=0.6, linewidth=1.2)
    ax.text(3.7, 0.655, 'Transfer\nthreshold\n(0.65)', fontsize=8, color=WARN_COLOR,
           ha='right', va='bottom')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.2, linewidth=1)

    ax.set_ylabel('AUROC')
    ax.set_title('Probe Generalization Across Distributions')
    ax.set_ylim(0.4, 0.92)
    ax.grid(True, axis='y', alpha=0.15, color=GRID_COLOR)

    savefig(fig, '05_generalization')


# ---------------------------------------------------------------------------
# Figure 6: Activation Space Projection
# ---------------------------------------------------------------------------

def fig_activation_space(clf, best_layer):
    print('Figure 6: Activation space')
    all_data = TRAIN + TEST
    all_X = np.stack([load_act(t['trajectory_id'], best_layer) for t in all_data])
    all_y = np.array([t['label'] for t in all_data])
    is_test = np.array([0] * len(TRAIN) + [1] * len(TEST))

    direction = clf.coef_[0]
    direction_norm = direction / np.linalg.norm(direction)

    proj_probe = all_X @ direction_norm
    residual = all_X - np.outer(proj_probe, direction_norm)
    pca = PCA(n_components=1)
    proj_pca = pca.fit_transform(residual).ravel()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot test points larger, train points smaller
    for is_t, marker, size, alpha in [(0, 'o', 25, 0.3), (1, 'D', 50, 0.8)]:
        mask = is_test == is_t
        for label, color, name in [(0, ADMIT_COLOR, 'Admit'), (1, FABRICATE_COLOR, 'Fabricate')]:
            submask = mask & (all_y == label)
            ax.scatter(proj_probe[submask], proj_pca[submask],
                      c=color, marker=marker, s=size, alpha=alpha,
                      edgecolors='white', linewidths=0.3)

    # Decision boundary
    bias = clf.intercept_[0]
    threshold_proj = -bias / np.linalg.norm(direction)
    ax.axvline(threshold_proj, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(threshold_proj + 0.05, ax.get_ylim()[1] * 0.9, 'Decision\nboundary',
           fontsize=8, color='gray', va='top')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=ADMIT_COLOR,
               markersize=8, label='Admit (test)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=FABRICATE_COLOR,
               markersize=8, label='Fabricate (test)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ADMIT_COLOR,
               markersize=6, alpha=0.5, label='Admit (train)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FABRICATE_COLOR,
               markersize=6, alpha=0.5, label='Fabricate (train)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_xlabel('Projection onto probe direction')
    ax.set_ylabel('First principal component (orthogonal to probe)')
    ax.set_title(f'Activation Space at Layer {best_layer}')
    ax.grid(True, alpha=0.1, color=GRID_COLOR)

    savefig(fig, '06_activation_space')


# ---------------------------------------------------------------------------
# Figure 7: Fabrication patterns heatmap
# ---------------------------------------------------------------------------

def fig_fabrication_heatmap():
    print('Figure 7: Fabrication heatmap')
    properties = ['band_gap', 'formation_energy_per_atom', 'density']
    prop_labels = ['Band gap', 'Formation energy', 'Density']

    matrix = np.zeros((5, 3))
    for ti in range(5):
        for pi, prop in enumerate(properties):
            sub = [t for t in BAL if t['template_index'] == ti and t['property'] == prop]
            if len(sub) > 0:
                matrix[ti, pi] = sum(t['label'] for t in sub) / len(sub)
            else:
                matrix[ti, pi] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(5):
        for j in range(3):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.6 or val < 0.2 else DARK
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                       fontweight='bold', fontsize=12, color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(prop_labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Template {i}' for i in range(5)])
    ax.set_title('Fabrication Rate by Template and Property\n(Balanced Prompt)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Fabrication Rate')

    fig.tight_layout()
    savefig(fig, '07_fabrication_heatmap')


# ---------------------------------------------------------------------------
# Figure 8: Summary dashboard
# ---------------------------------------------------------------------------

def fig_summary():
    print('Figure 8: Summary dashboard')
    fig = plt.figure(figsize=(14, 6))

    # Four key metrics as large numbers
    metrics = [
        ('Probe AUROC', '0.795', 'Layer 16, material-split', DARK),
        ('Fabrication\nReduction', '57%', 'Prompt injection', ACCENT),
        ('Cross-tool\nTransfer', '0.702', 'MP -> ChemDB', FABRICATE_COLOR),
        ('Accuracy\nPreserved', '100%', '0.0 pt degradation', ADMIT_COLOR),
    ]

    for i, (title, value, subtitle, color) in enumerate(metrics):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.text(0.5, 0.65, value, ha='center', va='center',
               fontsize=42, fontweight='bold', color=color,
               transform=ax.transAxes)
        ax.text(0.5, 0.92, title, ha='center', va='center',
               fontsize=12, fontweight='bold', color=DARK,
               transform=ax.transAxes)
        ax.text(0.5, 0.35, subtitle, ha='center', va='center',
               fontsize=10, color='gray', transform=ax.transAxes)

        # Decorative underline
        ax.plot([0.2, 0.8], [0.25, 0.25], color=color, linewidth=3,
               transform=ax.transAxes, clip_on=False)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.suptitle('Tool-Null Confabulation Probe: Key Results',
                fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, '08_summary_dashboard')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Generating figures...\n')

    test_aurocs, best_layer = fig_layer_sweep()
    fig_prompt_rates()
    clf = fig_roc(best_layer)
    fig_intervention()
    fig_transfer()
    fig_activation_space(clf, best_layer)
    fig_fabrication_heatmap()
    fig_summary()

    print(f'\nAll figures saved to {FIG_DIR}/')
    print(f'Files: {sorted(f.name for f in FIG_DIR.glob("*.png"))}')


if __name__ == '__main__':
    main()
