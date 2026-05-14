
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil

OUT_DIR = '/mnt/shared-workspace/shared'
os.makedirs(OUT_DIR, exist_ok=True)

N_ENVS = 100
N_NEURONS = 50
GRID_SIZE = 40  # 40x40 spatial bins

# ── Grid cell firing maps ─────────────────────────────────────────────────────
def make_grid_map(spacing, orientation, phase_x, phase_y, size=GRID_SIZE):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    # Three gratings at 60° apart
    rate = np.zeros((size, size))
    for k in range(3):
        angle = theta + k * np.pi / 3
        rate += np.cos(2 * np.pi / spacing * (X * np.cos(angle) + Y * np.sin(angle)) + phase_x)
    rate = (rate - rate.min()) / (rate.max() - rate.min() + 1e-8)
    return rate

# Example grid cell
grid_map_ex = make_grid_map(0.25, 15, 0.1, 0.2)

# ── Gridness scores ───────────────────────────────────────────────────────────
# Simulate gridness scores for N_NEURONS neurons
gridness_scores = np.random.normal(0.5, 0.4, N_NEURONS)
gridness_scores = np.clip(gridness_scores, -1, 1.5)
grid_cells = gridness_scores > 0.3
n_grid = grid_cells.sum()

# ── Place cell spatial information ────────────────────────────────────────────
spatial_info = np.abs(np.random.normal(1.5, 0.8, N_NEURONS))  # bits/spike
place_cells = spatial_info > 1.0
n_place = place_cells.sum()

# ── Head direction tuning ─────────────────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, 360)
# Von Mises fit for HD cells
preferred_dir = np.random.uniform(0, 2*np.pi, N_NEURONS)
kappa = np.random.exponential(2, N_NEURONS)  # concentration parameter
hd_tuning = np.array([np.exp(kappa[i] * np.cos(angles - preferred_dir[i])) for i in range(N_NEURONS)])
hd_tuning = hd_tuning / hd_tuning.max(axis=1, keepdims=True)
hd_cells = kappa > 1.5
n_hd = hd_cells.sum()

# ── Theta oscillation coupling ────────────────────────────────────────────────
# LFP theta phase (8 Hz)
t = np.linspace(0, 10, 10000)  # 10 seconds
lfp_theta = np.sin(2 * np.pi * 8 * t) + 0.3 * np.random.randn(len(t))
# Spike phases (preferentially at trough for place cells)
spike_phases = np.random.vonmises(np.pi, 2.0, 500)  # preferred phase = π
phase_locking = np.abs(np.mean(np.exp(1j * spike_phases)))  # mean resultant length

# ── Hippocampal-entorhinal connectivity ──────────────────────────────────────
hpc_activity = np.random.randn(N_NEURONS, N_ENVS)
ec_activity = np.random.randn(N_NEURONS, N_ENVS)
# Add correlation
ec_activity += 0.5 * hpc_activity + np.random.randn(N_NEURONS, N_ENVS) * 0.5
hpc_ec_corr = np.array([stats.pearsonr(hpc_activity[i], ec_activity[i])[0] for i in range(N_NEURONS)])

# ── Spatial coverage ──────────────────────────────────────────────────────────
# Fraction of environment covered by place fields
field_sizes = np.random.beta(2, 5, N_NEURONS) * 100  # % of environment
spatial_coverage = field_sizes.mean()

# ── Population vector correlation ────────────────────────────────────────────
# Correlation between population vectors across environments
pop_vec_corr = np.zeros((N_ENVS, N_ENVS))
for i in range(N_ENVS):
    for j in range(i, N_ENVS):
        r = stats.pearsonr(hpc_activity[:, i], hpc_activity[:, j])[0]
        pop_vec_corr[i, j] = pop_vec_corr[j, i] = r

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Spatial Neurogenomics Engine — Grid/Place Cells, 100 Environments × 50 Neurons',
             color='white', fontsize=15, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1: Grid cell firing map
ax = axes[0, 0]
im = ax.imshow(grid_map_ex, cmap='hot', aspect='auto', origin='lower')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Firing Rate (norm.)', color='#8b949e', fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
style_ax(ax, 'Grid Cell Firing Rate Map', 'X position', 'Y position')

# Panel 2: Gridness score distribution
ax = axes[0, 1]
ax.hist(gridness_scores, bins=25, color='#58a6ff', alpha=0.85, edgecolor='#0d1117', linewidth=0.5)
ax.axvline(0.3, color='#f78166', lw=2, linestyle='--', label=f'Grid cell threshold (n={n_grid})')
ax.axvline(gridness_scores.mean(), color='#3fb950', lw=2, linestyle='--',
           label=f'Mean={gridness_scores.mean():.2f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Gridness Score Distribution', 'Gridness Score', 'Count')

# Panel 3: Place cell spatial information
ax = axes[0, 2]
ax.hist(spatial_info, bins=25, color='#3fb950', alpha=0.85, edgecolor='#0d1117', linewidth=0.5)
ax.axvline(1.0, color='#f78166', lw=2, linestyle='--', label=f'Place cell threshold (n={n_place})')
ax.axvline(spatial_info.mean(), color='#ffa657', lw=2, linestyle='--',
           label=f'Mean={spatial_info.mean():.2f} bits/spike')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Place Cell Spatial Information (Skaggs)', 'Spatial Info (bits/spike)', 'Count')

# Panel 4: Head direction tuning
ax = axes[1, 0]
# Polar-like plot in Cartesian
example_hd = hd_tuning[np.argmax(kappa)]
ax.plot(np.rad2deg(angles), example_hd, color='#d2a8ff', lw=2, label='Best HD cell')
ax.fill_between(np.rad2deg(angles), 0, example_hd, alpha=0.3, color='#d2a8ff')
ax.axhline(0.5, color='#f78166', lw=1, linestyle='--', label='Half-max')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Head Direction Tuning (κ={kappa.max():.2f})', 'Direction (°)', 'Normalized Rate')

# Panel 5: Theta phase locking
ax = axes[1, 1]
ax.hist(spike_phases, bins=36, color='#ffa657', alpha=0.85, edgecolor='#0d1117', linewidth=0.3)
ax.axvline(np.pi, color='#f78166', lw=2, linestyle='--', label=f'Preferred phase (π)')
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], color='#8b949e')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Theta Phase Locking (MRL={phase_locking:.3f})', 'LFP Phase (rad)', 'Spike Count')

# Panel 6: Entorhinal-hippocampal correlation
ax = axes[1, 2]
ax.hist(hpc_ec_corr, bins=25, color='#79c0ff', alpha=0.85, edgecolor='#0d1117', linewidth=0.5)
ax.axvline(hpc_ec_corr.mean(), color='#f78166', lw=2, linestyle='--',
           label=f'Mean r={hpc_ec_corr.mean():.3f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Hippocampal-Entorhinal Correlation', 'Pearson r', 'Count')

# Panel 7: Spatial coverage
ax = axes[2, 0]
ax.hist(field_sizes, bins=25, color='#56d364', alpha=0.85, edgecolor='#0d1117', linewidth=0.5)
ax.axvline(spatial_coverage, color='#f78166', lw=2, linestyle='--',
           label=f'Mean={spatial_coverage:.1f}%')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Place Field Spatial Coverage', 'Coverage (%)', 'Count')

# Panel 8: Population vector correlation
ax = axes[2, 1]
im2 = ax.imshow(pop_vec_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
cb2 = fig.colorbar(im2, ax=ax, shrink=0.8, pad=0.02)
cb2.set_label('Pearson r', color='#8b949e', fontsize=8)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
style_ax(ax, 'Population Vector Correlation (Environments)', 'Environment j', 'Environment i')

# Panel 9: Summary
ax = axes[2, 2]
ax.set_facecolor('#161b22')
ax.axis('off')
summary_text = (
    f"  Spatial Neurogenomics Summary\n"
    f"  {'─'*32}\n"
    f"  Environments:          {N_ENVS}\n"
    f"  Neurons:               {N_NEURONS}\n"
    f"  Grid cells (>0.3):     {n_grid} ({100*n_grid/N_NEURONS:.1f}%)\n"
    f"  Mean gridness score:   {gridness_scores.mean():.3f}\n"
    f"  Place cells (>1 bit):  {n_place} ({100*n_place/N_NEURONS:.1f}%)\n"
    f"  Mean spatial info:     {spatial_info.mean():.3f} bits/spike\n"
    f"  HD cells (κ>1.5):      {n_hd} ({100*n_hd/N_NEURONS:.1f}%)\n"
    f"  Theta phase locking:   MRL={phase_locking:.3f}\n"
    f"  HPC-EC mean corr:      r={hpc_ec_corr.mean():.3f}\n"
    f"  Mean field coverage:   {spatial_coverage:.1f}%\n"
    f"  Mean pop vec corr:     {np.triu(pop_vec_corr, 1).mean():.3f}\n"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        color='#e6edf3', bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))
ax.set_title('Summary Statistics', color='white', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = f'{OUT_DIR}/spatial_neurogenomics_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_png}")

print("\n=== SpatialNeurogenomicsEngine Key Results ===")
print(f"N environments: {N_ENVS}, N neurons: {N_NEURONS}")
print(f"Grid cells (gridness>0.3): {n_grid} ({100*n_grid/N_NEURONS:.1f}%)")
print(f"Mean gridness score: {gridness_scores.mean():.4f}")
print(f"Place cells (spatial info>1 bit): {n_place} ({100*n_place/N_NEURONS:.1f}%)")
print(f"Mean spatial information: {spatial_info.mean():.4f} bits/spike")
print(f"HD cells (kappa>1.5): {n_hd} ({100*n_hd/N_NEURONS:.1f}%)")
print(f"Theta phase locking (MRL): {phase_locking:.4f}")
print(f"Mean HPC-EC correlation: r={hpc_ec_corr.mean():.4f}")
print(f"Mean spatial coverage: {spatial_coverage:.2f}%")
