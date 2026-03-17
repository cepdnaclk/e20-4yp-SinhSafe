import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOSS CURVE GENERATOR
# ==========================================
def plot_loss_curve(version_name, steps, train_loss, eval_loss):
    plt.figure(figsize=(8, 5))
    
    # Plotting the lines
    plt.plot(steps, train_loss, label='Training Loss', marker='o', color='#1f77b4', linewidth=2)
    plt.plot(steps, eval_loss, label='Evaluation Loss', marker='s', color='#d62728', linewidth=2)
    
    # Formatting (Using 'Steps' instead of 'Epochs' for LLM)
    plt.title(f'{version_name} - Training vs. Evaluation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure
    filename = f"{version_name.replace(' ', '_')}_Loss_Curve.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

# ==========================================
# 2. METRIC COMPARISON GENERATOR (BAR CHART)
# ==========================================
def plot_metric_comparison(versions, accuracy, precision, recall, f1_score):
    x = np.arange(len(versions))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Creating grouped bars
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, precision, width, label='Precision (W)', color='#ff7f0e')
    ax.bar(x + 0.5*width, recall, width, label='Recall (W)', color='#2ca02c')
    ax.bar(x + 1.5*width, f1_score, width, label='F1-Score (W)', color='#d62728')

    # Formatting
    ax.set_ylabel('Scores (%)', fontsize=12)
    ax.set_title('SinLLaMA Versions: Overall Metric Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=11)
    
    # Move legend outside to avoid covering the bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # SinLLaMA scored lower, so scaling Y-axis from 40 to 70
    ax.set_ylim([40, 70]) 

    # Save the figure
    plt.tight_layout()
    plt.savefig('SinLLaMA_Metrics_Comparison.png', dpi=300, bbox_inches="tight")
    print("Saved: SinLLaMA_Metrics_Comparison.png")
    plt.close()

# ==========================================
# 3. YOUR SINLLAMA DATA (Filtered for Eval Steps)
# ==========================================

steps_v1 = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
steps_v2_v3 = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
steps_v4 = [50, 100, 150, 200, 250, 300, 350, 400, 450]

# V1 (Triggers Early Stopping at Step 700 due to 3 consecutive increases)
plot_loss_curve("SinLLaMA Version 1", steps_v1, 
                [0.8635, 0.7494, 0.7220, 0.6813, 0.6272, 0.6080, 0.5841, 0.5449, 0.5045, 0.4687, 0.5092, 0.4296, 0.3990, 0.2965], 
                [0.8095, 0.7456, 0.7110, 0.6940, 0.6759, 0.6700, 0.6709, 0.6699, 0.6716, 0.6759, 0.6527, 0.6757, 0.6831, 0.7306])

# V2 
plot_loss_curve("SinLLaMA Version 2", steps_v2_v3, 
                [0.5452, 0.4810, 0.4711, 0.4657, 0.4208, 0.4185, 0.4217, 0.4317, 0.4092, 0.3792, 0.4081, 0.3987, 0.3865], 
                [0.5092, 0.4770, 0.4596, 0.4490, 0.4427, 0.4371, 0.4310, 0.4307, 0.4304, 0.4280, 0.4266, 0.4274, 0.4251])

# V3 (The best performing version - 65.66% F1)
plot_loss_curve("SinLLaMA Version 3", steps_v2_v3, 
                [0.5453, 0.4806, 0.4709, 0.4651, 0.4188, 0.4175, 0.4209, 0.4283, 0.4063, 0.3765, 0.4049, 0.3942, 0.3820], 
                [0.5092, 0.4767, 0.4589, 0.4486, 0.4426, 0.4360, 0.4303, 0.4305, 0.4295, 0.4269, 0.4254, 0.4259, 0.4251])

# V4 (Textbook Early Stopping trigger after Step 300)
plot_loss_curve("SinLLaMA Version 4", steps_v4, 
                [0.5142, 0.4632, 0.4208, 0.3905, 0.3661, 0.3346, 0.2957, 0.2362, 0.2099], 
                [0.4872, 0.4499, 0.4331, 0.4278, 0.4249, 0.4204, 0.4352, 0.4356, 0.4568])


# --- Metric Comparison Data ---
versions = ['V1', 'V2', 'V3', 'V4']
accuracy_data  = [48.76, 57.33, 66.34, 54.04]
precision_data = [56.49, 56.72, 65.20, 54.71]
recall_data    = [48.76, 57.33, 66.34, 54.04]
f1_data        = [42.93, 56.94, 65.66, 59.82] 

plot_metric_comparison(versions, accuracy_data, precision_data, recall_data, f1_data)
print("All SinLLaMA graphs successfully generated!")