from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import dill

class Muscle(Dataset):
  def __init__(self, train = True, transformX = None, transformY = None):
    # I have previously created a file named 500_train.csv using the file names. Here we will read in the csv to access data in google drive.
    # hayo: should this be 300_train.csv??
    validation_set_size = 0.2
    self.pixel_file = pd.read_csv('/Users/taliacho/Downloads/Ranger Lab/Bryan-Ranger/local_data/300_train.csv')
    self.transformX = transformX
    self.transformY = transformY
    self.train = train
    # Split the dataset to train and validation using sklearn function train_test_split
    self.train_data, self.validation_data = train_test_split(self.pixel_file,
                                                                test_size = validation_set_size,
                                                                random_state = 5)
  def __len__(self):
    if self.train:
      return len(self.train_data)
    return len(self.validation_data)

  def __getitem__(self, index):
    train_path = '/Users/taliacho/Downloads/Ranger Lab/Bryan-Ranger/local_data/train_data'
    if self.train:
      imx_name = os.path.join(train_path, self.train_data.iloc[index, 1])
      imy_name = os.path.join(train_path, self.train_data.iloc[index, 1].replace('.jpeg','_mask.jpg'))
    else:
      imx_name = os.path.join(train_path, self.validation_data.iloc[index, 1])
      imy_name = os.path.join(train_path, self.validation_data.iloc[index, 1].replace('.jpeg','_mask.jpg'))

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
        angle = random.choice([-30, -90, -60, -45 -15, 0, 15, 30, 45, 60, 90])
        imx = TF.rotate(imx, angle)
        imy = TF.rotate(imy, angle)

    # We will use resize, tensorlize, and normalize in the following cell
    if self.transformX :
      imx = self.transformX(imx)
      imy = self.transformY(imy)

    sample = {'image': imx, 'mask': imy}
    return sample