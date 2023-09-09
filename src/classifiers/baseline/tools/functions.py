import os
import cv2
import numpy as np


def load_images_from_folder(folder, max_images_per_class=100, flat=True):
    images = []
    labels = []
    class_id = 0

    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)

        if not os.path.isdir(subdir_path):
            continue

        image_count = 0

        for filename in os.listdir(subdir_path):
            if image_count >= max_images_per_class:
                break

            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is not None:
                if flat:
                    images.append(img.ravel())
                else:
                    images.append(img)

                labels.append(class_id)

                image_count += 1

        class_id += 1

    return np.array(images), np.array(labels)
