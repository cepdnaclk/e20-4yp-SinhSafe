import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# EXTRACTED DATA FROM LOGS (SinBERT)
# ==========================================
# 1. Performance Metrics (Percentages)
metrics = ['Precision', 'F1-Score']
v1_scores = [78.08, 77.86]  # From V1 Final Average Scores
v2_scores = [90.89, 90.71]  # From V2 Final Average Scores

# 2. Evaluation Loss Drop
loss_labels = ['V1 Dataset\n(Lowest Eval Loss)', 'V2 Dataset\n(Lowest Eval Loss)']
loss_values = [0.5213, 0.2704]  # V1 Fold 5 vs V2 Fold 1

# ==========================================
# PLOT 1: The Performance Leap (Grouped Bar Chart)
# ==========================================
plt.figure(figsize=(8, 6))

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of the bars

# Create bars
rects1 = plt.bar(x - width/2, v1_scores, width, label='V1 Dataset (6,075 docs)', color='#8E9BAE') # Muted gray-blue
rects2 = plt.bar(x + width/2, v2_scores, width, label='V2 Dataset (16,545 docs)', color='#1A5F7A') # Bold tech blue

# Add text, labels, and custom x-axis tick labels
plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
plt.title('SinBERT Performance: V1 vs. V2 Dataset', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, metrics, fontsize=12)
plt.ylim(0, 110) # Set limit higher so labels fit
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('performance_leap_chart.png', dpi=300, transparent=True)
print("✅ Saved 'performance_leap_chart.png'")
plt.close()

# ==========================================
# PLOT 2: Evaluation Loss Reduction
# ==========================================
plt.figure(figsize=(6, 5))

# Create bars
bars = plt.bar(loss_labels, loss_values, width=0.5, color=['#E67E22', '#27AE60']) # Orange to Green

plt.ylabel('Evaluation Loss', fontsize=12, fontweight='bold')
plt.title('Model Error Reduction (Overfitting Proof)', fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, 0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), 
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add a text box highlighting the 50% drop
plt.text(0.5, 0.6, "↓ ~48% Drop in Error Rate!", ha='center', va='center', 
         fontsize=12, fontweight='bold', color='red',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('eval_loss_drop.png', dpi=300, transparent=True)
print("✅ Saved 'eval_loss_drop.png'")
plt.close()