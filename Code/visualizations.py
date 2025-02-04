import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set the style using seaborn
sns.set_style("whitegrid")

# Strategy name mapping
strategy_names = {
    "lora": "LoRA",
    "ia3": "IA3", 
    "full_fine_tuning": "Full Model",
    "head_only": "Classification Head"
}

# Read the data
embed_results = pd.read_csv('Results/embedding_classification_results.csv')
finetune_results = pd.read_csv('Results/fine_tuning_classification_results.csv')

# Map strategy names
finetune_results['Strategy'] = finetune_results['Strategy'].map(strategy_names)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create color palette with truncated colormap
classifiers = embed_results['Classifier'].unique()
strategies = finetune_results['Strategy'].unique()
all_methods = list(classifiers) + list(strategies)
n_colors = len(all_methods)
colors = plt.cm.plasma(np.linspace(0, 0.9, n_colors))  # Truncated to 90%
color_dict = dict(zip(all_methods, colors))

# Plot embedding classification results
for classifier in classifiers:
    mask = embed_results['Classifier'] == classifier
    ax.scatter(embed_results[mask]['Accuracy'], 
              embed_results[mask]['EmbeddingModel'],
              marker='o', 
              s=100,
              color=color_dict[classifier],
              label=f'{classifier}')

# Plot fine-tuning classification results
for strategy in strategies:
    mask = finetune_results['Strategy'] == strategy
    ax.scatter(finetune_results[mask]['eval_Accuracy'],
              finetune_results[mask]['Model'],
              marker='s',
              s=100,
              color=color_dict[strategy],
              label=f'{strategy}')

# Add vertical grid lines
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Customize the plot
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.title('Classification Performance Comparison', fontsize=14, pad=20)

# Create legend with two rows
embed_legend = plt.legend(handles=[plt.scatter([], [], marker='o', color=color_dict[c], s=100, label=c) 
                                 for c in classifiers],
                         title='Embed then Classify',
                         bbox_to_anchor=(0.5, -0.15),
                         loc='upper center',
                         ncol=len(classifiers))
ax.add_artist(embed_legend)

finetune_legend = plt.legend(handles=[plt.scatter([], [], marker='s', color=color_dict[s], s=100, label=s) 
                                    for s in strategies],
                            title='Fine-Tune then Classify',
                            bbox_to_anchor=(0.5, -0.3),
                            loc='upper center',
                            ncol=len(strategies))

# Adjust layout to prevent legend cutoff
plt.subplots_adjust(bottom=0.25)

# Create the images directory if it doesn't exist
os.makedirs('Results/images', exist_ok=True)

# Save the plot
plt.savefig('Results/images/classification_comparison.png', dpi=900, bbox_inches='tight')
plt.close()