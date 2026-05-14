import os
from PIL import Image

# Path to your source folder
source_folder = "/Users/taliacho/Downloads/Bryan-Ranger/clinical_data/ABS_Ethiopia"
# Path to your destination folder
destination_folder = os.path.join(os.path.dirname(source_folder), "ABS_Ethiopia_bw")

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all files in the folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpeg", ".jpg")):
        file_path = os.path.join(source_folder, filename)
        img = Image.open(file_path)

        # Convert to black and white
        bw_img = img.convert("L")

        # Handle naming
        if filename.lower().endswith(".jpeg"):
            # Original image - keep same name
            new_name = filename
        elif filename.lower().endswith(".jpg"):
            # Mask image - rename to *_m.jpeg
            base_name = os.path.splitext(filename)[0]
            new_name = f"{base_name}_m.jpeg"

        # Save to new folder
        new_path = os.path.join(destination_folder, new_name)
        bw_img.save(new_path, "JPEG")

print(f"✅ All images processed and saved to: {destination_folder}")