import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOSS CURVE GENERATOR
# ==========================================
def plot_loss_curve(version_name, epochs, train_loss, eval_loss):
    plt.figure(figsize=(8, 5))
    
    # Plotting the lines
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='#1f77b4', linewidth=2)
    plt.plot(epochs, eval_loss, label='Evaluation Loss', marker='s', color='#d62728', linewidth=2)
    
    # Formatting
    plt.title(f'{version_name} - Average Cross-Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
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

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Creating grouped bars
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, precision, width, label='Precision (W)', color='#ff7f0e')
    ax.bar(x + 0.5*width, recall, width, label='Recall (W)', color='#2ca02c')
    ax.bar(x + 1.5*width, f1_score, width, label='F1-Score (W)', color='#d62728')

    # Formatting
    ax.set_ylabel('Scores (%)', fontsize=12)
    ax.set_title('SinBERT Versions: Overall Metric Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=11)
    
    # Move legend outside to avoid covering the bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Since metrics are around 75-80%, zooming the Y-axis makes the differences visible
    ax.set_ylim([70, 82]) 

    # Save the figure
    plt.tight_layout()
    plt.savefig('SinBERT_Metrics_Comparison.png', dpi=300, bbox_inches="tight")
    print("Saved: SinBERT_Metrics_Comparison.png")
    plt.close()

# ==========================================
# 3. YOUR SINBERT DATA (Pre-calculated Averages)
# ==========================================

# --- Loss Curve Data ---
epochs_4 = [1, 2, 3, 4]
epochs_8 = [1, 2, 3, 4, 5, 6, 7, 8]

# V1 Averages
plot_loss_curve("SinBERT Version 1", epochs_4, 
                [0.8269, 0.5267, 0.3932, 0.3133], 
                [0.6420, 0.5674, 0.5674, 0.5767])

# V2 Averages (Shows severe overfitting starting at Epoch 4)
plot_loss_curve("SinBERT Version 2", epochs_8, 
                [1.0281, 0.6708, 0.4822, 0.3632, 0.2796, 0.2146, 0.1768, 0.1530], 
                [0.8124, 0.5998, 0.5473, 0.5621, 0.5831, 0.6384, 0.6650, 0.6704])

# V3 Averages
plot_loss_curve("SinBERT Version 3", epochs_8, 
                [1.0306, 0.6696, 0.4826, 0.3660, 0.2798, 0.2196, 0.1790, 0.1558], 
                [0.8049, 0.6064, 0.5523, 0.5768, 0.6085, 0.6410, 0.6620, 0.6679])

# V4 Averages
plot_loss_curve("SinBERT Version 4", epochs_4, 
                [0.8482, 0.4929, 0.3237, 0.2229], 
                [0.6157, 0.5529, 0.6022, 0.6436])

# V5 Averages
plot_loss_curve("SinBERT Version 5", epochs_4, 
                [0.9054, 0.5573, 0.3992, 0.3120], 
                [0.6625, 0.5759, 0.5466, 0.5790])

# --- Metric Comparison Data ---
versions = ['V1', 'V2', 'V3', 'V4', 'V5']
accuracy_data  = [77.33, 78.09, 78.12, 77.86, 77.14]
precision_data = [77.37, 78.05, 78.13, 78.08, 77.15]
recall_data    = [77.33, 78.09, 78.12, 77.86, 77.14]
f1_data        = [77.30, 78.05, 77.07, 77.86, 77.09] 

plot_metric_comparison(versions, accuracy_data, precision_data, recall_data, f1_data)
print("All SinBERT graphs successfully generated!")