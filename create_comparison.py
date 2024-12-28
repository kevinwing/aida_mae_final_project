import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
models = ['YOLOv8 (Small)', 'YOLOv8 (Large)', 'YOLOv10 (Small)', 'YOLOv10 (Large)', 'SSD', 'Faster R-CNN']
precision = [49.4, 90.9, 51.4, 92.9, 0.03, 0]  # Precision percentages
recall = [49.4, 82.1, 46.0, 86.0, 0.0, 0]  # Recall percentages
map_50 = [45.3, 90.0, 47.5, 93.1, 0.015, 0]  # mAP50 percentages

# Bar width and x positions
x = np.arange(len(models))
bar_width = 0.2

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width, precision, bar_width, label='Precision (%)', color='blue')
ax.bar(x, recall, bar_width, label='Recall (%)', color='green')
ax.bar(x + bar_width, map_50, bar_width, label='mAP50 (%)', color='orange')

# Formatting the chart
ax.set_xlabel('Models')
ax.set_ylabel('Performance Metrics (%)')
ax.set_title('Comparison of Key Metrics Across Models')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the chart
plt.tight_layout()
plt.show()

