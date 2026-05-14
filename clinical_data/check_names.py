#folder has odd number of files... something is named incorrectly or duplicated 
import os

# Path to your image folder
folder_path = "/Users/taliacho/Downloads/Bryan-Ranger/clinical_data/ABS_all_bw"

# Get all files ending in .jpeg or .jpg
files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpeg", ".jpg"))]

# Separate originals and masks
originals = set()
masks = set()

for f in files:
    name, ext = os.path.splitext(f)
    if name.endswith("_m"):
        masks.add(name[:-2])  # remove '_m'
    else:
        originals.add(name)

# Find mismatches
missing_masks = sorted(originals - masks)
missing_originals = sorted(masks - originals)

# Display results
print("📁 Checking folder:", folder_path)
print("\n🧾 Summary:")
print(f"  Total originals: {len(originals)}")
print(f"  Total masks: {len(masks)}")

print("\n❌ Originals missing masks:")
for name in missing_masks:
    print(f"  {name}.jpeg")

print("\n❌ Masks missing originals:")
for name in missing_originals:
    print(f"  {name}_m.jpeg")

if not missing_masks and not missing_originals:
    print("\n✅ All image–mask pairs are consistent!")
import os

# Path to your image folder
folder_path = "path/to/your/folder"

# Get all files ending in .jpeg or .jpg
files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpeg", ".jpg"))]

# Separate originals and masks
originals = set()
masks = set()

for f in files:
    name, ext = os.path.splitext(f)
    if name.endswith("_m"):
        masks.add(name[:-2])  # remove '_m'
    else:
        originals.add(name)

# Find mismatches
missing_masks = sorted(originals - masks)
missing_originals = sorted(masks - originals)

# Display results
print("📁 Checking folder:", folder_path)
print("\n🧾 Summary:")
print(f"  Total originals: {len(originals)}")
print(f"  Total masks: {len(masks)}")

print("\n❌ Originals missing masks:")
for name in missing_masks:
    print(f"  {name}.jpeg")

print("\n❌ Masks missing originals:")
for name in missing_originals:
    print(f"  {name}_m.jpeg")

if not missing_masks and not missing_originals:
    print("\n✅ All image–mask pairs are consistent!")
