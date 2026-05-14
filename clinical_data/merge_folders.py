import os
import shutil

# Paths to your three source folders
folder1 = "/Users/taliacho/Downloads/Bryan-Ranger/clinical_data/ABS_BHW_bw"
folder2 = "/Users/taliacho/Downloads/Bryan-Ranger/clinical_data/ABS_Ethiopia_bw"
folder3 = "/Users/taliacho/Downloads/Bryan-Ranger/clinical_data/ABS_UMN_bw"

# Path to the destination folder
destination = os.path.join(os.path.dirname(folder1), "ABS_all_bw")

# Create destination folder if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Combine the folder paths into a list
source_folders = [folder1, folder2, folder3]

# Function to copy files safely
def copy_files(src_folder, dst_folder):
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        # Skip directories (only handle files)
        if os.path.isdir(src_path):
            continue

        # Handle duplicate filenames
        base, ext = os.path.splitext(filename)
        dst_path = os.path.join(dst_folder, filename)
        counter = 1
        while os.path.exists(dst_path):
            new_filename = f"{base}_{counter}{ext}"
            dst_path = os.path.join(dst_folder, new_filename)
            counter += 1

        # Copy file
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} → {dst_path}")

# Merge all folders
for folder in source_folders:
    if os.path.exists(folder):
        copy_files(folder, destination)
    else:
        print(f"⚠️ Skipped missing folder: {folder}")

print(f"\n✅ Merge complete! All files saved to: {destination}")