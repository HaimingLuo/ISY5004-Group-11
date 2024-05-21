import os
import imageio
import imgaug.augmenters as iaa

def load_and_augment_images(directory, output_directory, images_per_gesture=200):
    seq = iaa.Sequential([
        iaa.Cutout(nb_iterations=1, size=0.3, squared=False)  # Large area occlusion, size=0.3 indicates that the occlusion area accounts for 30% of the image size
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

input_dir = r'C:\Users\gymd1\Desktop\handpose_datasets_v2'
output_dir = r'C:\Users\gymd1\Desktop\handpose_datasets_v2_augmentedMask'

load_and_augment_images(input_dir, output_dir)
