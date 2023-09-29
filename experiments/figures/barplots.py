import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read data from the CSV
df = pd.read_csv('./barplots/benchmarks.csv')

# Create a new column combining Setting and Resolution
df['Setting-Resolution'] = df['Setting'] + '-' + df['Resolution'].astype(str)

# Create a 1x2 grid of subplots (1 row, 2 columns)
sns.set_context("talk", font_scale=1.1)  # Adjust the font scaling here

fig, ax = plt.subplots(1, 2, figsize=(20, 4))

# Time subplot using seaborn
sns.barplot(data=df, x='Model', y='Time', hue='Setting-Resolution', ax=ax[0], ci=None, dodge=True, order=df['Model'].unique())
ax[0].set_title('Time of Inference')
ax[0].tick_params(axis='x', rotation=20)
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place the legend to the right of the subplot

# GPU consumption subplot using seaborn
sns.barplot(data=df, x='Model', y='GPU', hue='Setting-Resolution', ax=ax[1], ci=None, dodge=True, order=df['Model'].unique())
ax[1].set_title('GPU Consumption')
ax[1].tick_params(axis='x', rotation=20)
ax[1].legend().set_visible(False)  # Remove the legend from the second subplot

# Adjust layout and save
plt.tight_layout()

save_dir = "./barplots/"
save_path = os.path.join(save_dir, 'benchmarks.svg')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(save_path, format='svg')
save_path = os.path.join(save_dir, 'benchmarks.png')
fig.savefig(save_path, format='png')
plt.close(fig)
