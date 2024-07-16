import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


class Muscle_test(Dataset):
    def __init__(self, transformX = None):
        self.pixel_file = pd.read_csv(os.path.join('/content/gdrive/MyDrive/Phantom Data/Rectus Abdominis/data/300_test.csv'))
        self.transformX = transformX

    def __len__(self):
        return len(self.pixel_file)

    def __getitem__(self, index):
        imx_name = os.path.join('/content/gdrive/MyDrive/Phantom Data/Rectus Abdominis/data/500_test_data', self.pixel_file.iloc[index, 1])

        imx = Image.open(imx_name)

        f_name = self.pixel_file.iloc[index, 1]

        if self.transformX :
            imx = self.transformX(imx)

        sample = {'image': imx, 'f_name': f_name}
        return sample

tx_X = transforms.Compose([transforms.Resize((256, 256)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])
test_data = Muscle_test(transformX = tx_X)
test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle = True)