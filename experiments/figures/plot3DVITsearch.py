import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('3DVit_128cf.csv')

grouped = df.groupby(['subvolume_size', 'patch_size', 'n_layers', 'd_model', 'd_ff', 'n_heads', 'd_encoder']).agg({'dice_val_avg': 'max'}).reset_index()
grouped['key'] = grouped.index + 1

# Normalize the dice_val_avg values to get colors
norm = plt.Normalize(grouped['dice_val_avg'].min(), grouped['dice_val_avg'].max())

# Truncate the inferno colormap to go from dark to orange
from matplotlib.colors import ListedColormap
inferno_truncated = ListedColormap(plt.cm.inferno(np.linspace(0, 0.8, 256)))  # Adjust the 0.8 if needed
colors = inferno_truncated(norm(grouped['dice_val_avg']))

# Create the horizontal bar plot
plt.figure(figsize=(5, 3.5))  # Adjust dimensions as needed
sns.barplot(x='dice_val_avg', y='key', data=grouped, palette=colors, saturation=1, orient='h')

# Set up the color bar and labels
sm = plt.cm.ScalarMappable(cmap=inferno_truncated, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('dice_val_avg', rotation=270, labelpad=15)
plt.title('Max dice_val_avg by Hyperparameters Group')
plt.ylabel('Hyperparameter Group Key')
plt.xlabel('dice_val_avg')

# Save the figure as an SVG
plt.tight_layout()

plt.savefig('3-1-1-3dviths.svg', format='svg')
plt.close()
