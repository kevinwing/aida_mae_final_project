import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import _utils
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassAccuracy
import json
from yolo_loader import *
import numpy as np


def save_checkpoint(model, optimizer, epoch, loss, output_dir, filename="checkpoint.pth"):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    print(f"Checkpoint saved: {filename}")


def save_training_history(history, output_dir, filename="training_history.json"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved: {filename}")


def evaluate(model, data_loader, device, dataset_type="Validation"):
    """
    Evaluate the model on the given dataset.
    Calculates loss, accuracy, mAP, mAP@50, and mAP@75.

    Args:
        model: The trained SSD model.
        data_loader: DataLoader for the evaluation dataset.
        device: The device to run the evaluation on.
        criterion: The loss function (e.g., MultiBoxLoss).
        dataset_type: Type of the dataset (e.g., "Validation" or "Test").

    Returns:
        Dictionary with evaluation metrics: loss, accuracy, mAP, mAP@50, and mAP@75.
    """
    total_loss = 0.0
    total_correct = 0
    total_labels = 0
    num_batches = 0

    # Initialize mAP metric
    mean_ap_metric = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5, 0.75])
    class_accuracy = MulticlassAccuracy(num_classes=model.head.classification_head.num_classes, average='macro')
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if not images or not targets:  # Skip empty batches
                print(f"Skipping empty batch at index {batch_idx}")
                continue

            # Prepare images and targets
            images = [torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0 for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()

            # Update mAP metric
            mean_ap_metric.update(outputs, targets)

            # Calculate accuracy (for classification)
            for output, target in zip(outputs, targets):
                preds = output["labels"]
                gt_labels = target["labels"]
                class_accuracy.update(preds, gt_labels)
 

            num_batches += 1
            if batch_idx % 10 == 0:
                print(f"{dataset_type} Batch [{batch_idx}/{len(data_loader)}] evaluated.")

    # Compute final metrics
    mean_ap = mean_ap_metric.compute()
    mean_ap_metric.reset()
    
    accuracy = class_accuracy.compute()
    class_accuracy.reset()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    print(f"{dataset_type} Evaluation Completed.")
    print(f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, mAP: {mean_ap['map']:.4f}, "
          f"mAP@50: {mean_ap['map_50']:.4f}, mAP@75: {mean_ap['map_75']:.4f}")

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "mAP": mean_ap["map"],
        "mAP@50": mean_ap["map_50"],
        "mAP@75": mean_ap["map_75"],
    }


def train_ssd(dataset_dir, target_size=(300, 300), num_classes=1, num_epochs=10, batch_size=8, learning_rate=0.001, output_dir="output_ssd"):
    # Load dataset
    dataset = YoloDataset(root=dataset_dir, target_size=target_size)

    # Split dataset into train, validation, and test sets
    dataset_size = len(dataset)
    test_size = int(0.1 * dataset_size)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)
    model.head.classification_head.num_classes = num_classes + 1  # Add background class

    # Adjust classification head
    in_channels = _utils.retrieve_out_channels(model.backbone, (target_size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Training history
    training_history = {
        "epochs": [],
        "training_loss": [],
        "validation_loss": [],
        "validation_mAP": [],
        "validation_mAP@50": [],
        "validation_mAP@75": []
    }

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (images, targets) in enumerate(train_loader):
            if not images or not targets:
                print(f"Skipping empty batch at index {batch_idx}")
                continue

            images = [torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0 for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"    Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {losses.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # Validation evaluation
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation")
        val_metrics = evaluate(model, val_loader, device, dataset_type="Validation")

        # Log training and validation metrics
        training_history["epochs"].append(epoch + 1)
        training_history["training_loss"].append(avg_train_loss)
        training_history["validation_loss"].append(val_metrics["avg_loss"])
        training_history["validation_mAP"].append(val_metrics["mAP"])
        training_history["validation_mAP@50"].append(val_metrics["mAP@50"])
        training_history["validation_mAP@75"].append(val_metrics["mAP@75"])

        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs}] Metrics:")
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_metrics['avg_loss']:.4f}, "
              f"mAP: {val_metrics['mAP']:.4f}, mAP@50: {val_metrics['mAP@50']:.4f}, mAP@75: {val_metrics['mAP@75']:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"ssd_checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, output_dir, filename=checkpoint_path)

        scheduler.step()

    # Save final model and training history
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "ssd300_vgg16_final.pth"))
    print("Final model saved.")
    save_training_history(training_history, output_dir)

    # Test Evaluation
    print("\nEvaluating on Test Set:")
    test_results = evaluate(model, test_loader, device, dataset_type="Test")

    # Save test results
    test_results_file = os.path.join(output_dir, "test_results.json")
    with open(test_results_file, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved: {test_results_file}")



if __name__ == "__main__":
    # Dataset directory
    dataset_dir = "./datasets/dataset"

    # Train SSD model
    train_ssd(dataset_dir=dataset_dir, target_size=(300, 300), num_classes=1, num_epochs=10, batch_size=8)

