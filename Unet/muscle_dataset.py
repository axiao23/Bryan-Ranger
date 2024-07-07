import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(self, train=True, transformX=None, transformY=None):
        # Reading the new CSV file for accessing data.
        self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')
        self.pixel_file['filename'] = self.pixel_file['filename'].astype(str)  # Ensure filenames are strings
        self.transformX = transformX
        self.transformY = transformY
        self.train = train

        # Split the dataset to train and validation using sklearn function train_test_split
        self.train_data, self.validation_data = train_test_split(self.pixel_file,
                                                                 test_size=0.2,  # replace `validation_set_size` with actual value
                                                                 random_state=5)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.validation_data)

    def __getitem__(self, index):
        train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UM_masked'

        if self.train:
            imx_name = os.path.join(train_path, str(self.train_data.iloc[index, 0]))
            imy_name = os.path.join(train_path, str(self.train_data.iloc[index, 0]).replace('.jpg', '_m.jpg'))
        else:
            imx_name = os.path.join(train_path, str(self.validation_data.iloc[index, 0]))
            imy_name = os.path.join(train_path, str(self.validation_data.iloc[index, 0]).replace('.jpg', '_m.jpg'))

        print(f"imx_name: {imx_name}, imy_name: {imy_name}")

        # Original image
        try:
            imx = Image.open(imx_name)
        except Exception as e:
            print(f"Error opening image {imx_name}: {e}")
            raise e

        # Mask for the image
        try:
            imy = Image.open(imy_name).convert('L')
        except Exception as e:
            print(f"Error opening mask {imy_name}: {e}")
            raise e

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

        # Applying the provided transformations
        if self.transformX:
            imx = self.transformX(imx)
        if self.transformY:
            imy = self.transformY(imy)

        sample = {'image': imx, 'mask': imy}
        return sample

# Example usage
transformX = None  # Replace with your transform
transformY = None  # Replace with your transform

train_loader = DataLoader(Muscle(train=True, transformX=transformX, transformY=transformY), batch_size=4, num_workers=0)

# Debugging the DataLoader
for ith_batch, sample_batched in enumerate(train_loader):
    print(ith_batch, sample_batched['image'].size(), sample_batched['mask'].size())
    if ith_batch >= 2:  # limit the number of batches to check
        break
