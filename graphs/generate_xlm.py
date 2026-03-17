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
    filename = f"{version_name.replace(' ', '_').replace('-', '')}_Loss_Curve.png"
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
    ax.set_title('XLM-RoBERTa Versions: Overall Metric Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=11)
    
    # Move legend outside to avoid covering the bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dynamic Y-axis limits based on data
    ax.set_ylim([70, 85]) 

    # Save the figure
    plt.tight_layout()
    plt.savefig('XLMR_Metrics_Comparison.png', dpi=300, bbox_inches="tight")
    print("Saved: XLMR_Metrics_Comparison.png")
    plt.close()

# ==========================================
# 3. YOUR XLM-R DATA (Pre-calculated Averages)
# ==========================================

epochs_5 = [1, 2, 3, 4, 5]
epochs_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epochs_8 = [1, 2, 3, 4, 5, 6, 7, 8]

# V1 Averages
plot_loss_curve("XLM-R Version 1", epochs_5, 
                [0.9764, 0.7192, 0.5711, 0.4732, 0.3987], 
                [0.8233, 0.6727, 0.5631, 0.5556, 0.5670])

# V2 Averages (Textbook Overfitting starting around Epoch 3)
plot_loss_curve("XLM-R Version 2", epochs_10, 
                [4.1982, 2.8628, 2.1259, 1.5769, 1.0835, 0.7290, 0.4898, 0.3084, 0.2142, 0.1091], 
                [0.8208, 0.6338, 0.6232, 0.6372, 0.7777, 0.7694, 0.9453, 1.1347, 1.2058, 1.2825])

# V3 Averages 
plot_loss_curve("XLM-R Version 3", epochs_8, 
                [2.1487, 1.7090, 1.4574, 1.3053, 1.1810, 1.0956, 1.0139, 0.9704], 
                [0.9420, 0.7601, 0.7127, 0.6974, 0.7025, 0.6909, 0.7152, 0.7122])


# --- Metric Comparison Data ---
versions = ['V1', 'V2', 'V3']
accuracy_data  = [77.27, 80.46, 79.31]
precision_data = [77.18, 80.46, 79.43]
recall_data    = [77.27, 80.46, 79.31]
f1_data        = [77.18, 80.41, 79.34] 

plot_metric_comparison(versions, accuracy_data, precision_data, recall_data, f1_data)
print("All XLM-RoBERTa graphs successfully generated!")