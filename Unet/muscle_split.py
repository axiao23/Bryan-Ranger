import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(
        self, 
        split="train",
        transformX=None,
        transformY=None,
        validation_set_size=0.2,
        test_set_size=0.1,
        # Base paths for your images/masks
        train_val_path='/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/abs_bw',
        test_path='/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/bw_infant_ar'
    ):
        """
        Args:
            split: Which subset of the data to load: 'train', 'validation', or 'test'.
            transformX: Transform pipeline for the input images.
            transformY: Transform pipeline for the mask images.
            validation_set_size: Fraction of total data reserved for validation.
            test_set_size: Fraction of total data reserved for testing.
            train_val_path: Base directory where training/validation images + masks are stored.
            test_path: Base directory where test images + masks are stored.
        """

        self.split = split
        self.transformX = transformX
        self.transformY = transformY

        # Store the base paths (where images/masks actually live on disk).
        self.train_val_path = train_val_path
        self.test_path = test_path

        # Read the CSV file that has all (or most) of your clinical data.
        self.pixel_file = pd.read_csv(
            '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv'
        )

        # -----------------------------
        # 1) Perform Train/Val/Test Split
        # -----------------------------
        # First, split into TRAIN (some fraction) and TEMP (the remainder).
        # TEMP will be further split into validation & test.
        temp_size = validation_set_size + test_set_size
        self.train_data, temp_data = train_test_split(
            self.pixel_file,
            test_size=temp_size,
            random_state=5
        )

        # Next, figure out how much of TEMP should be validation vs. test.
        # Example: if validation_set_size=0.1 and test_set_size=0.1,
        # then total temp_size=0.2. 
        # So validation should be 0.1 / 0.2 = 0.5 of the temp_data,
        # and test the other 0.5.
        validation_ratio_in_temp = validation_set_size / temp_size if temp_size > 0 else 0.0
        self.validation_data, self.test_data = train_test_split(
            temp_data,
            test_size=(1 - validation_ratio_in_temp),
            random_state=5
        )

        # -----------------------------
        # 2) Based on "split", choose which subset & path to use
        # -----------------------------
        if self.split == "train":
            self.data = self.train_data
            self.base_path = self.train_val_path
        elif self.split == "validation":
            self.data = self.validation_data
            self.base_path = self.train_val_path
        elif self.split == "test":
            self.data = self.test_data
            self.base_path = self.test_path
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Print some stats
        print(f"Total data size in CSV: {len(self.pixel_file)}")
        print(f"Train size: {len(self.train_data)} | Validation size: {len(self.validation_data)} | Test size: {len(self.test_data)}")
        print(f"Current split: {self.split} -> {len(self.data)} samples selected.")

    def __len__(self):
        """Return the number of samples in the chosen subset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return a single sample (image, mask)."""
        # -----------------------------
        # 1) Grab the filename from CSV
        # -----------------------------
        # Let's assume column 1 (index=1) in self.data has the filename.
        image_name = self.data.iloc[index, 1]

        # -----------------------------
        # 2) Construct full paths
        # -----------------------------
        if ".jpeg" in image_name.lower():
            imx_name = os.path.join(self.base_path, image_name)
            # Fix a small typo: ".jpeg" replaced with ".jpeg" in the string if needed
            # or image_name.replace('.jpeg', '_mask.jpg')
            imy_name = os.path.join(self.base_path, image_name.replace('.jpeg', '_mask.jpg'))
        else:
            imx_name = os.path.join(self.base_path, image_name)
            imy_name = os.path.join(self.base_path, image_name.replace('.jpg', '_m.jpg'))

        # -----------------------------
        # 3) Check if files exist
        # -----------------------------
        if not os.path.exists(imx_name) or not os.path.exists(imy_name):
            print(f"Missing file(s). Image: {imx_name}, Mask: {imy_name}")
            # Attempt to gracefully skip to next item
            return self.__getitem__((index + 1) % len(self.data))

        # -----------------------------
        # 4) Open image & mask
        # -----------------------------
        try:
            # Open the RGB (or grayscale) image
            imx = Image.open(imx_name)
            # Convert mask to L mode for a single channel
            imy = Image.open(imy_name).convert('L')
        except Exception as e:
            print(f"Error opening image or mask: {e}")
            # If there's an error, skip to next item
            return self.__getitem__((index + 1) % len(self.data))

        # -----------------------------
        # 5) Data Augmentations (Train only)
        # -----------------------------
        if self.split == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                imx = TF.hflip(imx)
                imy = TF.hflip(imy)
            
            # Random vertical flip
            if random.random() > 0.5:
                imx = TF.vflip(imx)
                imy = TF.vflip(imy)
            
            # Random rotation
            if random.random() > 0.8:
                angle = random.choice([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
                imx = TF.rotate(imx, angle)
                imy = TF.rotate(imy, angle)

        # -----------------------------
        # 6) Apply Transforms
        # -----------------------------
        if self.transformX:
            imx = self.transformX(imx)
        if self.transformY:
            imy = self.transformY(imy)

        # -----------------------------
        # 7) Return sample as a dict
        # -----------------------------
        sample = {
            'image': imx,
            'mask': imy
        }
        return sample