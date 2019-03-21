import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class DogsDataset(Dataset):
    """Dog breed identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            dataframe (pandas.core.frame.DataFrame): Pandas dataframe obtained
                by read_csv().
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.id[idx]) + ".jpg"
        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label]

def split_train_val(root_dir, train_size=0.8, train_transform=None, val_transform=None):
    dframe = pd.read_csv('labels.csv')
    labelnames = pd.read_csv('sample_submission.csv').keys()[1:]
    codes = range(len(labelnames))
    breed_to_code = dict(zip(labelnames, codes))
    dframe['target'] = [breed_to_code[x] for x in dframe.breed]

    cut = int(len(dframe)*train_size)
    train, val = np.split(dframe, [cut], axis=0)
    val = val.reset_index(drop=True)

    train_set = DogsDataset(root_dir, train, train_transform)
    val_set = DogsDataset(root_dir, val, val_transform)

    return train_set, val_set
