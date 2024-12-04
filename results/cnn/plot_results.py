import json
import matplotlib.pyplot as plt

# Load the JSON files
with open('cnn_training_history.json') as f:
    training_history = json.load(f)

with open('cnn_evaluation_results.json') as f:
    evaluation_results = json.load(f)

# Extract data for plotting
loss = training_history.get("loss", [])
val_loss = training_history.get("val_loss", [])
precision = training_history.get("class_output_accuracy", [])
recall = training_history.get("val_class_output_accuracy", [])
f1_score = training_history.get("class_output_loss", [])  # Proxy metric for F1-score
map_50 = evaluation_results[2]  # mAP@50
map_75 = evaluation_results[3]  # mAP@75

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(loss, label="Training Loss", color='blue')
plt.plot(val_loss, label="Validation Loss", linestyle="--", color='orange')
plt.title("Training and Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curves.png")
plt.show()

# Plot Precision and Recall Curves
plt.figure(figsize=(10, 6))
plt.plot(precision, label="Precision", color='green')
plt.plot(recall, label="Recall", linestyle="--", color='red')
plt.title("Precision and Recall Curves")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_curves.png")
plt.show()

# Plot F1-Score Curve (using class_output_loss as a proxy for demonstration)
plt.figure(figsize=(10, 6))
plt.plot(f1_score, label="F1-Score", color='purple')
plt.title("F1-Score Curve")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("f1_score_curve.png")
plt.show()

# Plot mAP@50 and mAP@75
plt.figure(figsize=(10, 6))
plt.bar(["mAP@50", "mAP@75"], [map_50, map_75], color=["blue", "orange"])
plt.title("mAP@50 and mAP@75")
plt.ylabel("Score")
plt.grid(axis="y")
plt.savefig("map_comparison.png")
plt.show()

