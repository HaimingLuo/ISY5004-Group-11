import os
from PIL import Image

input_folder = r'C:\Users\gymd1\Desktop\Template'
output_folder = r'C:\Users\gymd1\Desktop\TemplateCrop'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

failed_images = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((200, 200), Image.ANTIALIAS)
            img.save(os.path.join(output_folder, filename))
        except Exception as e:
            # Record the file name and exception information of the image that failed to be processed
            failed_images.append((filename, str(e)))

print('Picture resize completed!')

# Output a list of images that failed to process
if failed_images:
    print('The following image processing failed:')
    for fname, error in failed_images:
        print(f'{fname}: {error}')
else:
    print('All images processed successfully!')
