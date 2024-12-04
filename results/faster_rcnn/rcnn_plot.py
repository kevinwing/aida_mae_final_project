import torch
import matplotlib.pyplot as plt

# Load Faster R-CNN training history
history_path = "training_history.pth"  # Replace with your file path
history = torch.load(history_path)

# Example structure of history:
# {
#     'train_loss': [...],
#     'val_loss': [...],
#     'train_precision': [...],
#     'val_precision': [...],
#     'train_recall': [...],
#     'val_recall': [...],
#     'mAP@50': [...],
#     'mAP@75': [...]
# }

# Extract metrics
train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])
train_precision = history.get('train_precision', [])
val_precision = history.get('val_precision', [])
train_recall = history.get('train_recall', [])
val_recall = history.get('val_recall', [])
map_50 = history.get('mAP@50', [])
map_75 = history.get('mAP@75', [])

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", color='blue')
plt.plot(val_loss, label="Validation Loss", linestyle="--", color='orange')
plt.title("Training and Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("faster_rcnn_loss_curves.png")
plt.show()

# Plot Precision and Recall Curves
plt.figure(figsize=(10, 6))
plt.plot(train_precision, label="Training Precision", color='green')
plt.plot(val_precision, label="Validation Precision", linestyle="--", color='green')
plt.plot(train_recall, label="Training Recall", color='red')
plt.plot(val_recall, label="Validation Recall", linestyle="--", color='red')
plt.title("Precision and Recall Curves")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("faster_rcnn_precision_recall_curves.png")
plt.show()

# Plot mAP@50 and mAP@75 over epochs
plt.figure(figsize=(10, 6))
plt.plot(map_50, label="mAP@50", color='blue')
plt.plot(map_75, label="mAP@75", linestyle="--", color='orange')
plt.title("mAP@50 and mAP@75 Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("faster_rcnn_map_curves.png")
plt.show()

# Optional: Summarize results in a bar chart for the final epoch
if map_50 and map_75:
    plt.figure(figsize=(10, 6))
    plt.bar(["mAP@50", "mAP@75"], [map_50[-1], map_75[-1]], color=["blue", "orange"])
    plt.title("Final mAP@50 and mAP@75")
    plt.ylabel("Score")
    plt.grid(axis="y")
    plt.savefig("faster_rcnn_final_map.png")
    plt.show()

