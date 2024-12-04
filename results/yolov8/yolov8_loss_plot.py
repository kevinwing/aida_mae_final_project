import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "results.csv"  # Replace with the path to your results.csv file
results = pd.read_csv(csv_path)

# Strip leading and trailing spaces from column names
results.columns = results.columns.str.strip()

# Calculate training and validation losses
results['train_loss'] = results['train/box_loss'] + results['train/cls_loss'] + results['train/dfl_loss']
results['val_loss'] = results['val/box_loss'] + results['val/cls_loss'] + results['val/dfl_loss']

# Extract epochs and losses
epochs = results['epoch']
train_loss = results['train_loss']
val_loss = results['val_loss']

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--', color='orange')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("yolov8_training_validation_losses.png")
plt.show()

