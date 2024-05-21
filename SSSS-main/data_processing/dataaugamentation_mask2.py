import os
import random
import imageio
import imgaug.augmenters as iaa
import numpy as np
import cv2

def create_random_shape_mask(image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    shape_type = random.choice(["circle", "ellipse", "triangle", "parallelogram", "trapezoid", "polygon"])
    center = (random.randint(0, width), random.randint(0, height))
    max_radius = min(width, height) // 4

    if shape_type == "circle":
        radius = random.randint(max_radius // 2, max_radius)
        cv2.circle(mask, center, radius, 255, -1)
    elif shape_type == "ellipse":
        axes = (random.randint(max_radius // 2, max_radius), random.randint(max_radius // 2, max_radius))
        angle = random.randint(0, 360)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    elif shape_type == "triangle":
        pts = np.array([[center[0], center[1] - max_radius],
                        [center[0] - max_radius, center[1] + max_radius],
                        [center[0] + max_radius, center[1] + max_radius]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    elif shape_type == "parallelogram":
        dx = random.randint(max_radius // 2, max_radius)
        dy = random.randint(max_radius // 2, max_radius)
        pts = np.array([[center[0], center[1]],
                        [center[0] + dx, center[1] - dy],
                        [center[0] + dx, center[1] + dy],
                        [center[0], center[1] + dy * 2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    elif shape_type == "trapezoid":
        top_width = random.randint(max_radius // 2, max_radius)
        bottom_width = random.randint(max_radius // 2, max_radius)
        height = random.randint(max_radius // 2, max_radius)
        pts = np.array([[center[0] - top_width // 2, center[1] - height // 2],
                        [center[0] + top_width // 2, center[1] - height // 2],
                        [center[0] + bottom_width // 2, center[1] + height // 2],
                        [center[0] - bottom_width // 2, center[1] + height // 2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    elif shape_type == "polygon":
        num_sides = random.randint(5, 10)
        angle_step = 360 / num_sides
        radius = random.randint(max_radius // 2, max_radius)
        pts = []
        for i in range(num_sides):
            angle = np.deg2rad(i * angle_step)
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    return mask.astype(bool)

def apply_random_shape_mask(image, mask):
    image[mask] = 0
    return image

def load_and_augment_images(directory, output_directory, images_per_gesture=200):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(subdir, file)
                image = imageio.imread(filepath)
                for i in range(images_per_gesture):
                    mask = create_random_shape_mask(image.shape)
                    image_aug = apply_random_shape_mask(image.copy(), mask)
                    base_filename, file_extension = os.path.splitext(file)
                    aug_filename = f"{base_filename}_aug_{i+1}{file_extension}"
                    aug_filepath = os.path.join(output_directory, os.path.relpath(subdir, directory), aug_filename)
                    os.makedirs(os.path.dirname(aug_filepath), exist_ok=True)
                    imageio.imwrite(aug_filepath, image_aug)

input_dir = r'C:\Users\gymd1\Desktop\handpose_datasets_v2'
output_dir = r'C:\Users\gymd1\Desktop\handpose_datasets_v2_augmentedMaskShape1'

load_and_augment_images(input_dir, output_dir)



