import os
import random
from shutil import copy2

# Set the source and target directories\
source_directory = 'dataset3'
train_directory = 'train'
test_directory = 'test'

# Ensure the training and testing directories exist
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# Get all file paths in the source directory
file_paths = [os.path.join(source_directory, file) for file in os.listdir(source_directory)]
random.shuffle(file_paths)  # Randomly shuffle the file order

# Set the ratio for training and testing sets
train_ratio = 0.8
split_index = int(len(file_paths) * train_ratio)

# Split into training and testing sets
train_files = file_paths[:split_index]
test_files = file_paths[split_index:]

# Copy files to the training and testing directories
for file in train_files:
    copy2(file, train_directory)
for file in test_files:
    copy2(file, test_directory)

print("Dataset has been randomly shuffled and split into training and testing sets.")
