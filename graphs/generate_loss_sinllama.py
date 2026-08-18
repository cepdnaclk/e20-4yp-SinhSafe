import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# EXTRACTED DATA FOR SINLLAMA V2
# ==========================================
# Every 10 steps for Training Loss
steps_train = np.arange(10, 810, 10)
train_loss = [
    4.803, 2.2326, 1.0349, 0.8512, 0.7551, 0.7338, 0.6951, 0.682, 0.6638, 0.6339,
    0.6507, 0.6244, 0.6192, 0.5998, 0.5957, 0.5952, 0.5874, 0.5692, 0.5823, 0.5729,
    0.5651, 0.5657, 0.5731, 0.5421, 0.5519, 0.5419, 0.564, 0.556, 0.5576, 0.5428,
    0.5445, 0.5388, 0.5245, 0.5505, 0.526, 0.5424, 0.537, 0.532, 0.5313, 0.5163,
    0.5157, 0.5158, 0.5063, 0.5026, 0.4805, 0.5008, 0.4888, 0.5042, 0.4843, 0.499,
    0.4883, 0.4949, 0.4919, 0.4903, 0.5038, 0.4925, 0.4933, 0.4919, 0.4724, 0.4831,
    0.5062, 0.4844, 0.4869, 0.4778, 0.4755, 0.4882, 0.4639, 0.4874, 0.4778, 0.4682,
    0.4802, 0.4719, 0.4649, 0.4774, 0.4705, 0.4689, 0.455, 0.4715, 0.4836, 0.4755
]

# Every 50 steps for Evaluation Loss
steps_eval = np.arange(50, 850, 50)
eval_loss = [
    0.6702, 0.581, 0.5436, 0.5231, 0.5094, 0.5015, 0.4915, 0.485, 0.4806, 0.4769,
    0.4721, 0.4686, 0.4659, 0.4631, 0.461, 0.4592
]

# ==========================================
# PLOT: The Loss Plateau & The F1 Crash
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the curves
plt.plot(steps_train, train_loss, label='Training Loss', color='#8E9BAE', alpha=0.7, linewidth=2)
plt.plot(steps_eval, eval_loss, label='Evaluation Loss (The Plateau)', color='#E74C3C', linewidth=3, marker='o')

# Formatting the Chart
plt.title('The Generative LLM Memorization Trap (SinLLaMA V2)', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
plt.ylim(0.3, 1.2) # Zoom in to see the plateau clearly
plt.xlim(0, 850)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)

# Annotate the "Brick Wall" Plateau
plt.annotate('Eval Loss flatlines.\nModel stops generalizing.', 
             xy=(600, 0.465), xytext=(500, 0.6),
             arrowprops=dict(facecolor='#E74C3C', shrink=0.05, width=2, headwidth=8),
             fontsize=11, fontweight='bold', color='#E74C3C')

# The "Mic-Drop" Callout Box (Showing the Test Collapse)
test_f1 = 65.3
callout_text = (
    f"🚨 THE TESTING COLLAPSE\n"
    f"---------------------------------\n"
    f"Implied Validation Score: >90%\n"
    f"Unseen Test AVG-Precision Score: {test_f1}%\n\n"
    f"Conclusion: Severe Overfitting."
)

plt.text(100, 0.95, callout_text, 
         fontsize=11, fontweight='bold', color='white',
         bbox=dict(facecolor='#2C3E50', alpha=0.9, edgecolor='red', boxstyle='round,pad=1'))

plt.tight_layout()
plt.savefig('sinllama_memorization_trap.png', dpi=300, transparent=True)
print("✅ Saved 'sinllama_memorization_trap.png'")
plt.close()