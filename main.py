import os
import re

from PIL import Image
import pandas as pd
import numpy as np

from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import torch

from matplotlib import pyplot as plt

random_state = 42


class ImageData(Dataset):

    def __init__(self, data_path, img_folder, img_ids, labels, transform=None):
        self.img_path = os.path.join(data_path, img_folder)
        self.transform = transform
        self.img_ids = img_ids
        self.img_filenames = [f'{im}.jpg' for im in img_ids]
        self.labels = labels

    # Override to give PyTorch size of dataset
    def __len__(self):
            return len(self.img_filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filenames[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[index])
        return img, label


def get_train_val_test_samples(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, shuffle=True, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                      random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    df = pd.read_excel('products.xlsx')
    df.dropna(inplace=True, subset=['category', 'condition'])
    y = pd.get_dummies(df['category']).values
    ids = df['id'].values
    # work_df = df[['id', 'category', 'condition']]
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_samples(ids, y)
    dataset = ImageData('.', 'img_n', ids, y)
    img, label = dataset[0]
    plt.imshow(img)
    plt.show()
