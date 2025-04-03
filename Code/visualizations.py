import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_metric_comparison(embed_results, finetune_results, metric, output_path, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    strategy_names = {
        "lora": "LoRA",
        "ia3": "IA3",
        "full_fine_tuning": "Full Model",
        "head_only": "Classification Head"
    }

    finetune_results['Strategy'] = finetune_results['Strategy'].map(strategy_names)

    classifiers = embed_results['Classifier'].unique()
    strategies = finetune_results['Strategy'].unique()
    all_methods = list(classifiers) + list(strategies)
    n_colors = len(all_methods)
    colors = plt.cm.plasma(np.linspace(0, 0.9, n_colors))
    color_dict = dict(zip(all_methods, colors))

    for classifier in classifiers:
        mask = embed_results['Classifier'] == classifier
        ax.scatter(embed_results[mask][metric],
                  embed_results[mask]['EmbeddingModel'],
                  marker='o',
                  s=100,
                  color=color_dict[classifier],
                  label=classifier)

    for strategy in strategies:
        mask = finetune_results['Strategy'] == strategy
        ax.scatter(finetune_results[mask][f'eval_{metric}'],
                  finetune_results[mask]['Model'],
                  marker='s',
                  s=100,
                  color=color_dict[strategy],
                  label=strategy)

    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel(metric.replace('_', ' '), fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

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

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main execution
sns.set_style("whitegrid")

metrics = ['Accuracy', 'F1_Micro', 'F1_Macro', 'F1_Weighted']
dataset_names = ['Government_Documents', 'Hate_Speech', 'Liar2']




for metric in metrics:
    for dataset_name in dataset_names:
        os.makedirs(f'Results/{dataset_name}/images', exist_ok=True)
        embed_results = pd.read_csv(f'Results/{dataset_name}/embedding_classification_results.csv')
        finetune_results = pd.read_csv(f'Results/{dataset_name}/fine_tuning_classification_results.csv')
        plot_metric_comparison(
            embed_results,
            finetune_results,
            metric,
            f'Results/{dataset_name}/images/classification_{metric.lower()}_comparison.png',
            dataset_name
        )


#####################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_ablation_study(embed_results, finetune_results, metric, output_path, dataset_name):
    strategy_names = {
        "lora": "LoRA",
        "ia3": "IA3",
        "full_fine_tuning": "Full Model",
        "head_only": "Classification Head"
    }

    finetune_results['Strategy'] = finetune_results['Strategy'].map(strategy_names)

    # Filter for base/125 models
    base_models = [model for model in embed_results['EmbeddingModel'].unique()
                   if ('base' in model.lower() or '125' in model)]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    classifiers = embed_results['Classifier'].unique()
    strategies = finetune_results['Strategy'].unique()
    all_methods = list(classifiers) + list(strategies)
    n_colors = plt.cm.plasma(np.linspace(0, 0.9, len(all_methods)))
    color_dict = dict(zip(all_methods, n_colors))

    for idx, model in enumerate(base_models):
        ax = axes[idx]

        # Plot embedding results
        for classifier in classifiers:
            mask = (embed_results['Classifier'] == classifier) & \
                   (embed_results['EmbeddingModel'] == model)
            data = embed_results[mask].sort_values('TrainingDataPercent')
            ax.plot(data['TrainingDataPercent'], data[metric],
                    color=color_dict[classifier], marker='o', markersize=8)

        # Plot fine-tuning results with dashed lines
        for strategy in strategies:
            mask = (finetune_results['Strategy'] == strategy) & \
                   (finetune_results['Model'] == model)
            data = finetune_results[mask].sort_values('TrainingDataPercent')
            ax.plot(data['TrainingDataPercent'], data[f'eval_{metric}'],
                    color=color_dict[strategy], marker='s', markersize=8,
                    linestyle='--')  # Dashed line for fine-tuning

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Training Data %')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(model)

    # Remove extra subplots if any
    for idx in range(len(base_models), len(axes)):
        fig.delaxes(axes[idx])

    # Add legends
    embed_legend = fig.legend(handles=[plt.Line2D([], [], color=color_dict[c], marker='o',
                                                  linestyle='-', label=c, markersize=8)
                                       for c in classifiers],
                              title='Embed then Classify',
                              bbox_to_anchor=(0.5, -0.05),
                              loc='lower center',
                              ncol=len(classifiers))

    finetune_legend = fig.legend(handles=[plt.Line2D([], [], color=color_dict[s], marker='s',
                                                     linestyle='--', label=s, markersize=8)
                                          for s in strategies],
                                 title='Fine-Tune then Classify',
                                 bbox_to_anchor=(0.5, -0.1),
                                 loc='lower center',
                                 ncol=len(strategies))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main execution
sns.set_style("whitegrid")

metrics = ['Accuracy', 'F1_Micro', 'F1_Macro', 'F1_Weighted']
dataset_name = 'Government_Documents'
os.makedirs('Results/images', exist_ok=True)

# for metric in metrics:
#     embed_results = pd.read_csv(f'Results/{dataset_name}/embedding_classification_ablation.csv')
#     finetune_results = pd.read_csv(f'Results/{dataset_name}/fine_tuning_classification_ablation.csv')

#     plot_ablation_study(
#         embed_results,
#         finetune_results,
#         metric,
#         f'Results/{dataset_name}/images/ablation_{metric.lower()}_comparison.png',
#         dataset_name
#     )