#!/usr/bin/env python
"""
model_comparison_visualize.py

Generate visualizations comparing model performance with and without RAG
based on evaluation metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directory for output
os.makedirs("evaluation_results/figures", exist_ok=True)

# Load the data from CSV
# You can replace this with your CSV file path
data = pd.read_csv("evaluation_results/summary.csv")

# For better visualization, create a combined model+RAG column
data['Model_Config'] = data['Model'] + ' (' + data['RAG'].map({True: 'with RAG', False: 'without RAG'}) + ')'

# Set a nice color palette
colors = sns.color_palette("viridis", 6)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# ========================
# Figure 1: ROUGE & BLEU scores comparison across models
# ========================
plt.figure(figsize=(14, 8))

# Get the metrics we want to plot (excluding ExactMatch which is all zeros)
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']

# Create a grouped bar chart
bar_width = 0.15
x = np.arange(len(data['Model_Config'].unique()))

for i, metric in enumerate(metrics):
    offset = i * bar_width
    plt.bar(x + offset, data[metric], width=bar_width, label=metric, color=colors[i])

# Customize the plot
plt.xlabel('Model Configuration', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('NLG Metrics Comparison Across Models', fontsize=16, fontweight='bold')
plt.xticks(x + bar_width * (len(metrics) - 1) / 2, data['Model_Config'], rotation=45, ha='right')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('evaluation_results/figures/rouge_bleu_comparison.png', dpi=300, bbox_inches='tight')

# ========================
# Figure 2: Performance vs. Latency scatter plot
# ========================
plt.figure(figsize=(12, 8))

# Use ROUGE-L as the main performance metric
scatter = plt.scatter(data['Latency'], data['ROUGE-L'], 
                     s=200, # marker size
                     c=[colors[i] for i in range(len(data))], 
                     alpha=0.7)

# Add labels for each point
for i, txt in enumerate(data['Model_Config']):
    plt.annotate(txt, (data['Latency'][i], data['ROUGE-L'][i]),
                xytext=(10, 5), textcoords='offset points',
                fontsize=11)

# Add a diagonal line to show "ideal" performance (higher ROUGE-L, lower latency)
plt.plot([0, 4], [0, 0], 'k--', alpha=0.3)

# Customize the plot
plt.xlabel('Latency (seconds)', fontsize=14)
plt.ylabel('ROUGE-L Score', fontsize=14)
plt.title('Performance vs. Speed Trade-off', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, max(data['Latency']) * 1.1)
plt.ylim(0, max(data['ROUGE-L']) * 1.1)

# Highlight the "winner" with an annotation
best_idx = data['ROUGE-L'].idxmax()
plt.annotate('Best Performance', 
            xy=(data['Latency'][best_idx], data['ROUGE-L'][best_idx]),
            xytext=(data['Latency'][best_idx] - 1, data['ROUGE-L'][best_idx] - 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            fontsize=14, fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('evaluation_results/figures/performance_vs_latency.png', dpi=300, bbox_inches='tight')

# ========================
# Figure 3: RAG Impact Analysis (bonus visualization)
# ========================
plt.figure(figsize=(12, 8))

# Reshape data to analyze RAG impact
models = data['Model'].unique()
metrics_to_plot = ['ROUGE-1', 'ROUGE-L', 'BLEU']

# Calculate RAG improvement for each model and metric
rag_impact = pd.DataFrame()

for model in models:
    model_data = data[data['Model'] == model]
    no_rag = model_data[model_data['RAG'] == False]
    with_rag = model_data[model_data['RAG'] == True]
    
    for metric in metrics_to_plot:
        # Calculate absolute difference
        diff = with_rag[metric].values[0] - no_rag[metric].values[0]
        rag_impact = pd.concat([rag_impact, pd.DataFrame({
            'Model': [model],
            'Metric': [metric],
            'RAG_Impact': [diff]
        })])

# Plot the heatmap
rag_impact_pivot = rag_impact.pivot(index='Model', columns='Metric', values='RAG_Impact')
plt.figure(figsize=(10, 6))
sns.heatmap(rag_impact_pivot, annot=True, cmap='RdBu_r', center=0, fmt='.3f', linewidths=.5)
plt.title('Impact of RAG on Model Performance (Difference)', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save the figure
plt.savefig('evaluation_results/figures/rag_impact_heatmap.png', dpi=300, bbox_inches='tight')

# ========================
# Print insights from the data analysis
# ========================
print("\nAnalysis Results:")
print("=" * 50)

# Identify best model overall
best_model_idx = data['ROUGE-L'].idxmax()
best_model = data.iloc[best_model_idx]
print(f"Best performing model (by ROUGE-L): {best_model['Model']} with RAG={best_model['RAG']}")
print(f"Achieved ROUGE-L: {best_model['ROUGE-L']:.4f}, BLEU: {best_model['BLEU']:.4f}")

# Identify fastest model
fastest_model_idx = data['Latency'].idxmin()
fastest_model = data.iloc[fastest_model_idx]
print(f"Fastest model: {fastest_model['Model']} with RAG={fastest_model['RAG']}")
print(f"Latency: {fastest_model['Latency']:.4f} seconds")

# Calculate average impact of RAG
rag_impact_avg = rag_impact.groupby('Model')['RAG_Impact'].mean()
print("\nAverage impact of RAG by model:")
for model in models:
    impact = rag_impact_avg[model]
    direction = "improved" if impact > 0 else "decreased"
    print(f"{model}: RAG {direction} performance by {abs(impact):.4f} on average")

print("\nFigures saved to evaluation_results/figures/")

