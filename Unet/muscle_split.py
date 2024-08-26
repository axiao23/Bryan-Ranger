import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Muscle(Dataset):
    def __init__(self, split='train', transformX=None, transformY=None):
        
        # only clinical or phantom 
        # self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv')
        # self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/300_train.csv')

        # # clinical and phantom datasets train + test 
        train_val_file_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UMN_train.csv'
        test_file_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/inf_abs.csv'
        train_val_data = pd.read_csv(train_val_file_path)
        self.test_data = pd.read_csv(test_file_path)
        
        # combine clinical and phantom datasets 
        

        self.transformX = transformX
        self.transformY = transformY
        self.split = split

        # only clinical or phantom 
        # train_val_data, self.test_data = train_test_split(self.pixel_file, test_size=0.2, random_state=5)
        # self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.25, random_state=5)  # 0.25 * 0.8 = 0.2
        # print(f"Total data size: {len(train_val_data)}")
        # if self.split in ['train', 'validation']:
        #     print(f"Training set size: {len(self.train_data)}")
        #     print(f"Validation set size: {len(self.validation_data)}")
        # elif self.split == 'test':
        #     print(f"Test set size: {len(self.test_data)}")

        # clinical and phantom datasets train + test 
        print(f"Total data size: {len(train_val_data)}")
        if self.split in ['train', 'validation']:
            self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.2, random_state=5)
            print(f"Training set size: {len(self.train_data)}")
            print(f"Validation set size: {len(self.validation_data)}")
        elif self.split == 'test':
            print(f"Test set size: {len(self.test_data)}")

        # combine clinical and phantom datasets 
        # train_val_data, self.test_data = train_test_split(self.pixel_file, test_size=0.2, random_state=5)
        # self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.25, random_state=5)  # 0.25 * 0.8 = 0.2

        # print(f"Total data size: {len(train_val_data)}")
        # if self.split in ['train', 'validation']:
        #     print(f"Training set size: {len(self.train_data)}")
        #     print(f"Validation set size: {len(self.validation_data)}")
        # elif self.split == 'test':
        #     print(f"Test set size: {len(self.test_data)}")

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'validation':
            return len(self.validation_data)
        else:  # self.split == 'test'
            return len(self.test_data)

    def __getitem__(self, index):
        test_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/bw_infant_ar'
        train_val_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/ABS'
        
        # # train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/local_data/train_data'
        # train_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/ABS'
        
        if self.split == 'train':
            data = self.train_data
            base_path =  train_val_path
        elif self.split == 'validation':
            data = self.validation_data
            base_path =  train_val_path
        else:  # self.split == 'test'
            data = self.test_data
            base_path = test_path 
            
        # new_images_path = '/Users/taliacho/Downloads/Ranger/Bryan-Ranger/clinical_data/UM_masked'  # Directory containing the new images
        # new_images = []
        # for filename in os.listdir(new_images_path):
        #     if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file extensions if needed
        #         img_path = os.path.join(new_images_path, filename)
        #         new_images.append(img_path)
        #     if len(new_images) == 60:
        #         break
        # # Combine the new images with the existing test_data
        # base_path.extend(new_images)
    
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
