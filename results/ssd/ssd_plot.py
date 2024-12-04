import json
import matplotlib.pyplot as plt

# Load the training history JSON file
with open('training_history.json') as f:
    training_history = json.load(f)

# Extract relevant data
epochs = training_history["epochs"]
metrics = training_history["training_metrics"]

# Initialize lists for plotting
training_loss = []
validation_loss = []
validation_map = []
validation_map_50 = []
validation_map_75 = []

# Populate the lists from the JSON data
for epoch in epochs:
    epoch_metrics = metrics[str(epoch)]
    training_loss.append(epoch_metrics["training_loss"])
    validation_loss.append(epoch_metrics["validation_loss"])
    validation_map.append(epoch_metrics["validation_mAP"])
    validation_map_50.append(epoch_metrics["validation_mAP@50"])
    validation_map_75.append(epoch_metrics["validation_mAP@75"])

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label="Training Loss", color='blue')
plt.plot(epochs, validation_loss, label="Validation Loss", linestyle="--", color='orange')
plt.title("Training and Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("ssd_training_validation_losses.png")
plt.show()

# Plot mAP Metrics (mAP, mAP@50, mAP@75)
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_map, label="Validation mAP", color='green')
plt.plot(epochs, validation_map_50, label="Validation mAP@50", linestyle="--", color='purple')
plt.plot(epochs, validation_map_75, label="Validation mAP@75", linestyle=":", color='red')
plt.title("Validation mAP Metrics")
plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.legend()
plt.grid(True)
plt.savefig("ssd_validation_map_metrics.png")
plt.show()

