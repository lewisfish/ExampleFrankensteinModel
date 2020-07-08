import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class HouseDataset(Dataset):
    """docstring for HouseDataset"""
    def __init__(self, root, file, transform=None):
        super(HouseDataset, self).__init__()
        self.transform = transform
        self.root = Path(root)
        self.df = self._load_house_attributes(file)
        self.images = self._load_house_images()

        self.X = self._process_house_attributes()

    def __getitem__(self, idx):

        image = self.images[idx]
        # convert to tensor with order [c, h, w]
        image = torch.as_tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        attrs = self.X[idx]
        attrs = torch.as_tensor(attrs, dtype=torch.float)

        # seperate price out
        target = self.X[idx][-1]
        target = torch.as_tensor(target, dtype=torch.float)

        sample = (image, attrs, target)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(list(self.X))

    def _load_house_attributes(self, file):
        cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
        df = pd.read_csv(self.root / file, sep=" ", header=None, names=cols)

        zipcodes = df["zipcode"].value_counts().keys().tolist()
        counts = df["zipcode"].value_counts().tolist()

        for zipcode, count in zip(zipcodes, counts):
            if count < 25:
                idxs = df[df["zipcode"] == zipcode].index
                df.drop(idxs, inplace=True)

        return df

    def _process_house_attributes(self):

        self.maxPrice = self.df["price"].max()
        self.df["price"] = self.df["price"] / self.maxPrice

        continous = ["bedrooms", "bathrooms", "area"]

        cs = MinMaxScaler()
        dfCnt = cs.fit_transform(self.df[continous])

        zipBinarizer = LabelBinarizer().fit(self.df["zipcode"])
        dfCat = zipBinarizer.transform(self.df["zipcode"])

        X = np.hstack([dfCat, dfCnt])

        return X

    def _load_house_images(self):
        # initialize our images array (i.e., the house images themselves)
        images = []
        # loop over the indexes of the houses
        for i in self.df.index.values:
            # find the four images for the house and sort the file paths,
            # ensuring the four are always in the *same order*
            basePath = str(self.root / f"{i+1}_*")
            housePaths = sorted(list(glob.glob(basePath)))

            # initialize our list of input images along with the output image
            # after *combining* the four input images
            inputImages = []
            outputImage = np.zeros((64, 64, 3), dtype="uint8")
            # loop over the input house paths
            for housePath in housePaths:
                # load the input image, resize it to be 32 32, and then
                # update the list of input images
                image = cv2.imread(housePath)
                image = cv2.resize(image, (32, 32))
                inputImages.append(image)
            # tile the four input images in the output image such the first
            # image goes in the top-right corner, the second image in the
            # top-left corner, the third image in the bottom-right corner,
            # and the final image in the bottom-left corner
            outputImage[0:32, 0:32] = inputImages[0]
            outputImage[0:32, 32:64] = inputImages[1]
            outputImage[32:64, 32:64] = inputImages[2]
            outputImage[32:64, 0:32] = inputImages[3]
            # add the tiled image to our set of images the network will be
            # trained on
            images.append(outputImage)
        # return our set of images
        return np.array(images) / 255.0
