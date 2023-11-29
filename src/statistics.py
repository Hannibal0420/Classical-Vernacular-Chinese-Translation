import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Creating a DataFrame from the provided data
data = [
    {"Mean Perplexity": 7.14, "Steps": 25, "Batch": 4, "Rank": 4, "LoRA Alpha": 32},
    {"Mean Perplexity": 18.26, "Steps": 250, "Batch": 4, "Rank": 4, "LoRA Alpha": 32},
    {"Mean Perplexity": 19.83, "Steps": 250, "Batch": 4, "Rank": 4, "LoRA Alpha": 8},
    {"Mean Perplexity": 23.26, "Steps": 750, "Batch": 4, "Rank": 8, "LoRA Alpha": 32},
    {"Mean Perplexity": 4.79, "Steps": 750, "Batch": 8, "Rank": 4, "LoRA Alpha": 16},
    {"Mean Perplexity": 9.75, "Steps": 850, "Batch": 8, "Rank": 4, "LoRA Alpha": 16},
    {"Mean Perplexity": 8.57, "Steps": 475, "Batch": 16, "Rank": 4, "LoRA Alpha": 16},
    {"Mean Perplexity": 10.78, "Steps": 575, "Batch": 16, "Rank": 4, "LoRA Alpha": 16},
    {"Mean Perplexity": 10.90, "Steps": 675, "Batch": 16, "Rank": 4, "LoRA Alpha": 16},
    {"Mean Perplexity": 15.2, "Steps": 775, "Batch": 16, "Rank": 4, "LoRA Alpha": 16}
]

df = pd.DataFrame(data)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
basic_stats = df.describe()
correlation_matrix = df.corr()
print(basic_stats)
print(correlation_matrix)


# Normalize LoRA Alpha for dot sizes
loras = df['LoRA Alpha'].unique()
size_scale = {lora: (loras.tolist().index(lora) + 1) * 50 for lora in loras}

# Plotting
fig, ax = plt.subplots()

# Generate different shades of blue
batches = df['Batch'].unique()
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(batches)))  # Adjust 0.3 and 0.9 for color range

# Iterate over different batch sizes to plot with different shades of blue
for i, batch in enumerate(batches):
    batch_df = df[df['Batch'] == batch]
    ax.scatter(batch_df["Steps"], batch_df["Mean Perplexity"], s=batch_df["LoRA Alpha"]*10, color=colors[i], label=f'Batch {batch}')

# Customizing the plot
ax.set_xlabel("Training Steps")
ax.set_ylabel("Mean Perplexity")
ax.set_title("Distribution of Validation Loss")
ax.grid(True)
plt.legend()

# Show plot
plt.show()
