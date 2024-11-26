import os

def yolo_to_tf(yolo_label_file, image_width, image_height):
    """
    Convert YOLO bounding boxes to TensorFlow/Keras format.
    Args:
        yolo_label_file (str): Path to the YOLO label file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    Returns:
        list: Converted bounding boxes in TensorFlow format [class_id, xmin, ymin, xmax, ymax].
    """
    tf_boxes = []
    with open(yolo_label_file, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            xmin = (x_center - width / 2) * image_width
            ymin = (y_center - height / 2) * image_height
            xmax = (x_center + width / 2) * image_width
            ymax = (y_center + height / 2) * image_height
            tf_boxes.append([int(class_id), xmin, ymin, xmax, ymax])
    return tf_boxes

def convert_dataset(input_dir, output_dir, image_width, image_height):
    """
    Convert YOLO dataset to TensorFlow/Keras-compatible format.
    Args:
        input_dir (str): Path to the YOLO dataset directory.
        output_dir (str): Path to save the converted dataset.
        image_width (int): Width of the images.
        image_height (int): Height of the images.
    """
    for split in ["train", "val", "test"]:
        # Paths for images and labels
        image_input_dir = os.path.join(input_dir, "images", split)
        label_input_dir = os.path.join(input_dir, "labels", split)
        image_output_dir = os.path.join(output_dir, split, "images")
        label_output_dir = os.path.join(output_dir, split, "labels")

        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        # Copy images
        for image_file in os.listdir(image_input_dir):
            input_path = os.path.join(image_input_dir, image_file)
            output_path = os.path.join(image_output_dir, image_file)
            os.system(f"cp {input_path} {output_path}")

        # Convert and save labels
        for label_file in os.listdir(label_input_dir):
            if label_file.endswith(".txt"):
                yolo_path = os.path.join(label_input_dir, label_file)
                tf_boxes = yolo_to_tf(yolo_path, image_width, image_height)
                output_path = os.path.join(label_output_dir, label_file)
                with open(output_path, "w") as f:
                    for box in tf_boxes:
                        f.write(" ".join(map(str, box)) + "\n")

if __name__ == '__main__':
    # Example usage
    convert_dataset(
        input_dir="/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split",
        output_dir="/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset",
        image_width=1024,
        image_height=1024
    )

