import matplotlib.pyplot as plt
import numpy as np

def plot_production_loss(model_name, epochs, train_loss, eval_loss, save_name):
    plt.figure(figsize=(9, 6))
    
    # Plotting with professional styling
    plt.plot(epochs, train_loss, label='Avg Training Loss', marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.plot(epochs, eval_loss, label='Avg Evaluation Loss', marker='s', color='#d62728', linewidth=2.5, markersize=8)
    
    # Adding vertical line for best epoch (visual aid for Viva)
    best_epoch = epochs[np.argmin(eval_loss)]
    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    plt.title(f'{model_name} Production Model - Average K-Fold Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Successfully generated: {save_name}")
    plt.close()

# ==========================================
# DATA PROCESSING (Averaged across 5 Folds)
# ==========================================

# XLM-RoBERTa (8 Epochs)
xlm_epochs = [1, 2, 3, 4, 5, 6, 7, 8]
xlm_train = [3.4319, 1.9114, 1.3686, 1.0440, 0.8062, 0.6181, 0.4731, 0.3818]
xlm_eval  = [0.5514, 0.4313, 0.3970, 0.4080, 0.4557, 0.4419, 0.5169, 0.5224]

# SinBERT (4 Epochs)
bert_epochs = [1, 2, 3, 4]
bert_train = [0.6358, 0.2313, 0.1245, 0.0702]
bert_eval  = [0.3270, 0.3101, 0.4098, 0.4374]

# Generate Graphs
plot_production_loss("XLM-RoBERTa", xlm_epochs, xlm_train, xlm_eval, "XLM_Best_Version_Loss_Curve.png")
plot_production_loss("SinBERT", bert_epochs, bert_train, bert_eval, "SinBERT_Best_Version_Loss_Curve.png")