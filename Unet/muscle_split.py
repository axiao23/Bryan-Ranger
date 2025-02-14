import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(self, split = "train",transformX=None,transformY=None,validation_set_size=0.1,test_set_size=0.1):
        
        # Clinical data csv
        self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')
        self.transformX = transformX
        self.transformY = transformY
        self.split = split
        self.validation_set_size = validation_set_size
        self.test_set_size = test_set_size

        # Split the data into train, validation, and test sets
        self.train_data, temp_data = train_test_split(self.pixel_file, test_size=self.validation_set_size + self.test_set_size, random_state=5)
        
        # Now split the temp_data into validation and test sets
        validation_size = self.validation_set_size / (self.validation_set_size + self.test_set_size)
        self.validation_data, self.test_data = train_test_split(temp_data, test_size=(1 - validation_size), random_state=5)

         # Handle data selection based on the split
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "validation":
            self.data = self.validation_data
        elif self.split == "test":
            self.data = self.test_data
        else:
            raise ValueError(f"Unknown split: {self.split}")
 
    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'validation':
            return len(self.validation_data)
        else:  # self.split == 'test'
            return len(self.test_data)

    def __getitem__(self, index):
        train_val_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/abs_bw'
        
         # Handle data selection based on the split
        if self.split == "train":
            self.data = self.train_val_data
            self.base_path = self.base_data_path  # path for training/validation data
        elif self.split == "validation":
            self.data = self.validation_data
            self.base_path = self.base_data_path  # path for training/validation data
        elif self.split == "test":
            self.data = self.test_data
            self.base_path = self.test_data_path  # path for test data
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        
         # Handle data selection based on the split
        if self.split == "train":
            self.data = self.train_val_path
            self.base_path = os.path.join(self.base_data_path, "train")
        elif self.split == "validation":
            self.data = self.validation_data
            self.base_path = os.path.join(self.base_data_path, "validation")
        elif self.split == "test":
            self.data = self.test_data
            self.base_path = os.path.join(self.base_data_path, "test")
        else:
            raise ValueError(f"Unknown split: {self.split}")
        

        if self.split == 'train':
            data = self.train_data
            base_path =  train_val_path
        elif self.split == 'validation':
            data = self.validation_data
            base_path =  train_val_path
        else:  # self.split == 'test'
            data = self.test_data
            base_path = test_path 
            
    
        image_name = data.iloc[index, 1]
        if ".jpeg" in image_name:
            imx_name = os.path.join(base_path, image_name)
            imy_name = os.path.join(base_path, image_name.replace('.jepg', '_mask.jpg'))
        else:
            imx_name = os.path.join(base_path, image_name)
            imy_name = os.path.join(base_path, image_name.replace('.jpg', '_m.jpg'))

        if not os.path.exists(imx_name) or not os.path.exists(imy_name):
            print(f"Image or mask file does not exist: {imx_name} or {imy_name}")
            return self.__getitem__((index + 1) % len(data))
                 
        if not os.path.exists(imx_name):
            print(f"Image file does not exist: {imx_name}")
        if not os.path.exists(imy_name):
            print(f"Mask file does not exist: {imy_name}")

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
