# Description: Create more classes for the Porsche dataset
import os
import shutil

# Set the source path and target path
src_path = "C:/Users/phili/.keras/datasets/resized_DVM/Porsche"
dst_path = "C:/Users/phili/.keras/datasets/resized_DVM/Porsche_more_classes"

# Create the target directory if it doesn't exist
os.makedirs(dst_path, exist_ok=True)

# Iterate through the folders and subfolders in the source directory
for root, dirs, files in os.walk(src_path):
    for dir_name in dirs:
        if dir_name.isdigit():  # Year folder
            model = os.path.basename(root)  # Get the model name
            for color_dir in os.listdir(os.path.join(root, dir_name)):
                src_color_dir = os.path.join(root, dir_name, color_dir)
                dst_model_year_dir = os.path.join(dst_path, f"{model}_{dir_name}", color_dir)
                # Copy color folder to the target directory under the combined model_year folder
                shutil.copytree(src_color_dir, dst_model_year_dir)
        elif os.path.basename(root) == "Porsche":  # Top-level model folders
            # Copy the entire model folder to the target directory if it doesn't have any subfolders
            if not any(os.path.isdir(os.path.join(root, dir_name, d)) for d in os.listdir(os.path.join(root, dir_name))):
                shutil.copytree(os.path.join(root, dir_name), os.path.join(dst_path, dir_name))