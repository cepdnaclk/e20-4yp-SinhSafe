import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# EXTRACTED DATA (SinBERT, XLM-R, SinLLaMA)
# ==========================================
models = ['SinBERT', 'XLM-RoBERTa', 'SinLLaMA (8B)']

# Average F1-Scores (%)
v1_f1_scores = [78.1, 80.5, 55.2]
v2_f1_scores = [90.89, 87.21, 65.29]

# Lowest Evaluation Loss
v1_eval_loss = [0.5213, 0.5445, 0.4251]
v2_eval_loss = [0.2704, 0.3401, 0.4592]

# ==========================================
# PLOT 1: The F1-Score Leap (Combined 3 Models)
# ==========================================
plt.figure(figsize=(10, 6))

x = np.arange(len(models))  # Label locations
width = 0.35  # Width of the bars

# Create bars
rects1 = plt.bar(x - width/2, v1_f1_scores, width, label='V1 Dataset (6,075 docs)', color='#8E9BAE')
rects2 = plt.bar(x + width/2, v2_f1_scores, width, label='V2 Dataset (16,545 docs)', color='#1A5F7A')

# Formatting
plt.ylabel('AVG-Precision (%)', fontsize=12, fontweight='bold')
plt.title('Architectural Showdown: V1 vs. V2 Dataset Performance', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, models, fontsize=12, fontweight='bold')
plt.ylim(0, 110) # Set higher to fit labels
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Data labels function
def autolabel_f1(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel_f1(rects1)
autolabel_f1(rects2)

plt.tight_layout()
plt.savefig('final_f1_leap.png', dpi=300, transparent=True)
print("✅ Saved 'final_f1_leap.png'")
plt.close()

# ==========================================
# PLOT 2: Evaluation Loss
# ==========================================
plt.figure(figsize=(10, 6))

# Create bars
rects3 = plt.bar(x - width/2, v1_eval_loss, width, label='V1 Eval Loss', color='#E67E22') # Orange
rects4 = plt.bar(x + width/2, v2_eval_loss, width, label='V2 Eval Loss', color='#27AE60') # Green

# Formatting
plt.ylabel('Lowest Evaluation Loss', fontsize=12, fontweight='bold')
plt.title('Error Rate Comparison (Overfitting Threshold)', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, models, fontsize=12, fontweight='bold')
plt.ylim(0, 0.8)
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Data labels function
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
plt.savefig('final_eval_loss.png', dpi=300, transparent=True)
print("✅ Saved 'final_eval_loss.png'")
plt.close()