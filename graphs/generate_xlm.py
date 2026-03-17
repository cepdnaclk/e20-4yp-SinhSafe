import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# EXTRACTED DATA (SinBERT & XLM-RoBERTa)
# ==========================================
models = ['SinBERT', 'XLM-RoBERTa']

# Average F1-Scores
v1_f1_scores = [77.86, 80.41]
v2_f1_scores = [90.71, 86.86]

# Lowest Evaluation Loss 
v1_eval_loss = [0.5213, 0.5445]
v2_eval_loss = [0.2704, 0.3401]

# ==========================================
# PLOT 1: The F1-Score Leap (Combined)
# ==========================================
plt.figure(figsize=(9, 6))

x = np.arange(len(models))  # Label locations
width = 0.35  # Width of the bars

# Create bars
rects1 = plt.bar(x - width/2, v1_f1_scores, width, label='V1 Dataset (6,075 docs)', color='#8E9BAE')
rects2 = plt.bar(x + width/2, v2_f1_scores, width, label='V2 Dataset (16,545 docs)', color='#1A5F7A')

# Formatting
plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
plt.title('Ensemble Components: V1 vs. V2 Dataset Performance', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, models, fontsize=12, fontweight='bold')
plt.ylim(0, 105) 
plt.legend(loc='upper left', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Data labels
def autolabel_f1(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel_f1(rects1)
autolabel_f1(rects2)

plt.tight_layout()
plt.savefig('combined_f1_leap.png', dpi=300, transparent=True)
print("✅ Saved 'combined_f1_leap.png'")
plt.close()

# ==========================================
# PLOT 2: Evaluation Loss Reduction (Combined)
# ==========================================
plt.figure(figsize=(9, 6))

# Create bars
rects3 = plt.bar(x - width/2, v1_eval_loss, width, label='V1 Eval Loss', color='#E67E22') # Orange
rects4 = plt.bar(x + width/2, v2_eval_loss, width, label='V2 Eval Loss', color='#27AE60') # Green

# Formatting
plt.ylabel('Lowest Evaluation Loss', fontsize=12, fontweight='bold')
plt.title('Error Rate Reduction Across Architectures', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, models, fontsize=12, fontweight='bold')
plt.ylim(0, 0.7)
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Data labels
def autolabel_loss(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel_loss(rects3)
autolabel_loss(rects4)

plt.tight_layout()
plt.savefig('combined_eval_loss.png', dpi=300, transparent=True)
print("✅ Saved 'combined_eval_loss.png'")
plt.close()