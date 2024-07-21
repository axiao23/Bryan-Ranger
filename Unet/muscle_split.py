import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(self, split='train', transformX=None, transformY=None):
        self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')
        self.transformX = transformX
        self.transformY = transformY
        self.split = split

        # Split the data into train, validation, and test sets
        train_val_data, self.test_data = train_test_split(self.pixel_file, test_size=0.2, random_state=5)
        self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.25, random_state=5)  # 0.25 * 0.8 = 0.2

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'validation':
            return len(self.validation_data)
        else:  # self.split == 'test'
            return len(self.test_data)

    def __getitem__(self, index):
        train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UM_masked'

        if self.split == 'train':
            data = self.train_data
        elif self.split == 'validation':
            data = self.validation_data
        else:  # self.split == 'test'
            data = self.test_data

        imx_name = os.path.join(train_path, data.iloc[index, 1])
        imy_name = os.path.join(train_path, data.iloc[index, 1].replace('.jpg', '_m.jpg'))

        try:
            # Original image
            imx = Image.open(imx_name)

            # Mask for the image
            imy = Image.open(imy_name).convert('L')
        except Exception as e:
            print(f"Error opening image: {e}")
            raise e

        if self.split == 'train':
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

        # We will use resize, tensorize, and normalize in the following cell
        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)

        sample = {'image': imx, 'mask': imy}
        return sample
