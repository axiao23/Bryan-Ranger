import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import skimage.morphology as mo
from skimage import io, color #Scikit-Image
from PIL import Image # Pillow
import cv2
import os
import random
import torch 
from torch.utils.data  import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import re

class Muscle(Dataset):
    def __init__(self, train=True, transformX=None, transformY=None):
        # Load the CSV file
        # self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/300_train.csv')   
        self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')

        # Strip leading and trailing spaces from the second column
        self.pixel_file.iloc[:, 1] = self.pixel_file.iloc[:, 1].str.strip()

        # Verify if file names are correct
        print("Sample file names after stripping spaces:")
        print(self.pixel_file.iloc[:, 1].head())

        # Check for invalid file names (optional)
        pattern = re.compile(r'^[\w,\s-]+\.[A-Za-z]{3,4}$')
        invalid_files = self.pixel_file[~self.pixel_file.iloc[:, 1].apply(lambda x: bool(pattern.match(x)))]
        if not invalid_files.empty:
            print("Invalid file names found:")
            print(invalid_files)

        # Split the dataset to train and validation using sklearn function train_test_split
        validation_set_size = 0.2
        self.train_data, self.validation_data = train_test_split(self.pixel_file,
                                                                 test_size=validation_set_size,
                                                                 random_state=5)
        self.transformX = transformX
        self.transformY = transformY
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.validation_data)

    def __getitem__(self, index):
        # train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/train_data'
        train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UM_masked'
        if self.train:
            imx_name = os.path.join(train_path, self.train_data.iloc[index, 1])
            imy_name = os.path.join(train_path, self.train_data.iloc[index, 1].replace('.jpg', '_m.jpg'))
        else:
            imx_name = os.path.join(train_path, self.validation_data.iloc[index, 1])
            imy_name = os.path.join(train_path, self.validation_data.iloc[index, 1].replace('.jpg', '_m.jpg'))

        try:
            # Check if files exist
            if not os.path.exists(imx_name):
                raise FileNotFoundError(f"File not found: {imx_name}")
            if not os.path.exists(imy_name):
                raise FileNotFoundError(f"File not found: {imy_name}")

            # original image
            imx = Image.open(imx_name)

            # mask for the image
            imy = Image.open(imy_name).convert('L')

            if self.train:
                # Random horizontal flipping
                if random.random() > 0.5:
                    imx = TF.hflip(imx)
                    imy = TF.hflip(imy)

                # Random vertical flipping
                if random.random() > 0.5:
                    imx = TF.vflip(imx)
                    imy = TF.vflip(imy)

                # Random rotation
                if random.random() > 0.8:
                    angle = random.choice([-30, -90, -60, -45, -15, 0, 15, 30, 45, 60, 90])
                    imx = TF.rotate(imx, angle)
                    imy = TF.rotate(imy, angle)

            # Apply transformations if defined
            if self.transformX:
                imx = self.transformX(imx)
                imy = self.transformY(imy)

            sample = {'image': imx, 'mask': imy}
            return sample

        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            return None

# Define transforms
transformX = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformY = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create the dataset and DataLoader
train_dataset = Muscle(train=True, transformX=transformX, transformY=transformY)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, timeout=60)  # Start with num_workers=0 for debugging

# Iterate through the DataLoader
for i, data in enumerate(train_loader):
    if data is not None:
        print(f"Batch {i} loaded successfully")
        # Process your data
    else:
        print(f"Batch {i} contains None")



