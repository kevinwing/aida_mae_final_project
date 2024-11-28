import albumentations as A
import cv2 as cv
import os
import matplotlib.pyplot as plt

MAX_SIZE = 1024

# define albumentations pipline
transform = A.Compose([
    A.LongestMaxSize(max_size=MAX_SIZE, interpolation=cv.INTER_LINEAR),
    A.PadIfNeeded(min_height=MAX_SIZE, min_width=MAX_SIZE, border_mode=cv.BORDER_CONSTANT, value=(0, 0, 0)),
], bbox_params=A.BboxParams(format='pascal_voc'))


def normalize_image(image, bboxes, class_lbls, size=1024):
    class_lbls = [0]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_lbls)

    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    return transformed_image, transformed_bboxes


def parse_labels(label_path):
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            # temp = line.split()[1:]
            temp = [float(i) for i in line.split()[1:]]
            temp.append('0')
            bboxes.append(temp)
    return bboxes


def process_directory(image_dir, label_dir, new_image_dir, new_label_dir):
    # class_lbls = [0]
    # class_lbls.append(0)
    # print(class_lbls)
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith('.jpg'):
            img_path = os.path.join(image_dir, file_name)
            lbl_path = os.path.join(label_dir, file_name.rsplit('.', 1)[0] + '.txt')

            if not os.path.exists(img_path):
                print(f"No image found: {img_path}")
                exit(1)

            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if not os.path.exists(lbl_path):
                print(f"No label file found: {lbl_path}")
                exit(1)

            bboxes = parse_labels(lbl_path)

            transformed = transform(image=img, bboxes=bboxes)

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            new_img_path = os.path.join(new_image_dir, file_name)
            cv.imwrite(new_img_path, transformed_image)

            with open(new_img_path, 'w') as f:
                for bbox in bboxes:
                    line = f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                    # print(line)
                    f.write(line)
            # print('\n')
            # print(lbl_path)
            # print(bboxes)

    # image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    # label_paths = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]

    # for entry in image_paths:
        # img_path = os.path.join(image_dir, entry)
        # img = cv.imread(img_path)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # if img is not None:
            # print(f"Loaded: {entry}, Shape: {img.shape}")


if __name__ == '__main__':
    base_dir = '/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset'
    new_base_dir = '/run/media/krw/PortableSSD/fall_2024_augmented_dataset/normalized_tf_dataset'
    os.makedirs(new_base_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(base_dir, split, 'images')
        # print(image_dir)
        labels_dir = os.path.join(base_dir, split, 'labels')
        # print(labels_dir)
        new_image_dir = os.path.join(new_base_dir, split, 'images')
        new_labels_dir = os.path.join(new_base_dir, split, 'labels')
        os.makedirs(new_image_dir, exist_ok=True)
        os.makedirs(new_labels_dir, exist_ok=True)
        process_directory(image_dir, labels_dir, new_image_dir, new_labels_dir)
