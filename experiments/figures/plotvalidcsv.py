import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load the data from CSV
df = pd.read_csv('validation_subv128.csv')  # Replace with the name of your CSV file

# Assign unique labels for each (Prelim) 3DVit entry
def create_label(row):
    if row['Model'] == '(Prelim) 3DVit':
        return '(Prelim) 3DVit_' + str(int(row.name)-1)
    else:
        return row['Model']

df['Plot_ID'] = df.apply(create_label, axis=1)

# Normalize the dice_val_avg values to get colors
norm = plt.Normalize(df['Dice Val Average'].min(), df['Dice Val Average'].max())

# Truncate the inferno colormap to go from dark to orange
inferno_truncated = ListedColormap(plt.cm.inferno(np.linspace(0, 0.8, 256)))  # Adjust the 0.8 if needed
colors = inferno_truncated(norm(df['Dice Val Average']))

# Create the horizontal bar plot
plt.figure(figsize=(8, 2))  # Adjust dimensions as needed
sns.barplot(x='Dice Val Average', y='Plot_ID', data=df, palette=colors, saturation=1, orient='h')

# Adjusting the y-tick labels to display just the model names
labels = df['Plot_ID']
plt.yticks(plt.yticks()[0], labels)

# Set up the color bar and labels
sm = plt.cm.ScalarMappable(cmap=inferno_truncated, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Dice Val Average', rotation=270, labelpad=15)
plt.title('Dice Val Average by Model')
plt.ylabel('Model')
plt.xlabel('Dice Val Average')

# Save the figure as an SVG
plt.tight_layout()
plt.savefig('barplot_distinct_output.svg', format='svg')
plt.savefig('barplot_distinct_output.png', format='png')
plt.show()  # Optionally show the plot
