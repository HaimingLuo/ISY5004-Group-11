import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

def load_and_augment_images(directory, output_directory, images_per_gesture=200):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% chance of horizontal flip
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-25, 25)
        ),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(subdir, file)
                image = imageio.imread(filepath)
                for i in range(images_per_gesture):
                    image_aug = seq(image=image)
                    base_filename, file_extension = os.path.splitext(file)
                    aug_filename = f"{base_filename}_aug_{i+1}{file_extension}"
                    aug_filepath = os.path.join(output_directory, os.path.relpath(subdir, directory), aug_filename)
                    os.makedirs(os.path.dirname(aug_filepath), exist_ok=True)
                    imageio.imwrite(aug_filepath, image_aug)

input_dir = r'C:\Users\gymd1\Desktop\2024.5.6'
output_dir = r'C:\Users\gymd1\Desktop\2024.5.6_augmentedNew'

load_and_augment_images(input_dir, output_dir)




