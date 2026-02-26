import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- AESTHETICS SETUP ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
colors = {
    'mlp': '#3b82f6',      # Bright Blue
    'ens': '#8b5cf6',      # Purple
    'gru': '#f59e0b',      # Orange
    'lnn': '#ef4444',      # Red
    'hyb': '#10b981'       # Emerald Green (Ours)
}
labels = {
    'mlp': 'MLP (Baseline)', 
    'ens': 'Ensemble', 
    'gru': 'GRU (History)', 
    'lnn': 'LNN (Physics)', 
    'hyb': 'Hybrid (Ours)'
}
linewidth = 2.5

print("Loading data...")
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

# Helper: Smoothing function for aesthetic lines
def smooth(y, box_pts=5):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

# ==========================================
# 1. ROLLOUT PLOTS (The Main Result)
# ==========================================
envs = list(results.keys())
models = ['mlp', 'ens', 'gru', 'lnn', 'hyb']

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for i, env in enumerate(envs):
    ax = axes[i]
    runs = results[env]
    
    # Find min length across all runs to crop
    min_len = 1000
    for r in runs:
        min_len = min(min_len, len(r['traj']['gt']))
    
    for m in models:
        # Collect Error Trajectories
        errors = []
        for r in runs:
            gt = np.array(r['traj']['gt'])[:min_len]
            pred = np.array(r['traj'][m])[:min_len]
            # Log Squared Error (Better for visualization than raw MSE)
            mse = (gt - pred)**2
            errors.append(mse)
            
        errors = np.array(errors) # (Seeds, Time)
        
        # Statistics
        mean = np.mean(errors, axis=0)
        # Standard Error of Mean (95% Confidence Interval approx)
        sem = np.std(errors, axis=0) / np.sqrt(len(errors))
        
        # Smoothing
        x = np.arange(len(mean))
        
        # Plot
        ax.plot(x, mean, color=colors[m], label=labels[m] if i==0 else "", linewidth=linewidth)
        ax.fill_between(x, mean - sem, mean + sem, color=colors[m], alpha=0.2) # Darker alpha

    ax.set_title(f"{env.capitalize()}", fontweight='bold')
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Squared Error (Log Scale)")
    ax.set_yscale('symlog', linthresh=1e-2) # Good hybrid scale
    ax.grid(True, which="both", ls="-", alpha=0.3)

# Global Legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("paper_rollout_results.png", dpi=300, bbox_inches='tight')
print("✅ Saved paper_rollout_results.png")

# ==========================================
# 2. BIFURCATION (Safety)
# ==========================================
unstable_envs = ['pendulum', 'cartpole', 'acrobot']
data = []

for env in unstable_envs:
    if env in results:
        runs = results[env]
        for m in ['mlp', 'ens', 'hyb']:
            vals = [r['bif'][m] for r in runs]
            for v in vals:
                data.append({'Environment': env.capitalize(), 'Model': labels[m], 'Uncertainty': v})

if data:
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(data)
    
    # Barplot with BLACK EDGES so MLP (0.0) is visible as a flat line
    sns.barplot(
        data=df, 
        x='Environment', 
        y='Uncertainty', 
        hue='Model', 
        palette=[colors['mlp'], colors['ens'], colors['hyb']],
        edgecolor="black", # <--- This fixes the "Invisible White" issue
        linewidth=1.5,
        err_kws={'linewidth': 2, 'color': 'black'} # Thicker error bars
    )
    
    plt.title("Safety Check: Uncertainty at Unstable Equilibrium", fontweight='bold')
    plt.ylabel("Predicted Variance ($\sigma^2$)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title=None)
    
    plt.tight_layout()
    plt.savefig("paper_bifurcation.png", dpi=300)
    print("✅ Saved paper_bifurcation.png")
