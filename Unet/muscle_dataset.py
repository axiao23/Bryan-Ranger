import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(self, train=True, transformX=None, transformY=None):
        self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')
        self.transformX = transformX
        self.transformY = transformY
        self.train = train

        self.train_data, self.validation_data = train_test_split(self.pixel_file,
                                                                 test_size=0.2,
                                                                 random_state=5)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.validation_data)

    def __getitem__(self, index):
        train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UM_masked'

        if self.train:
            imx_name = os.path.join(train_path, self.train_data.iloc[index, 1])
            imy_name = os.path.join(train_path, self.train_data.iloc[index, 1].replace('.jpg', '_m.jpg'))
        else:
            imx_name = os.path.join(train_path, self.validation_data.iloc[index, 1])
            imy_name = os.path.join(train_path, self.validation_data.iloc[index, 1].replace('.jpg', '_m.jpg'))

        try:
            # original image
            imx = Image.open(imx_name)

            # mask for the image
            imy = Image.open(imy_name).convert('L')
        except Exception as e:
            print(f"Error opening image: {e}")
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

        # We will use resize, tensorize, and normalize in the following cell
        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)

        sample = {'image': imx, 'mask': imy}
        return sample
