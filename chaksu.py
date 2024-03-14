import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset
import h5py


class Chaksu_Classification(Dataset):
    def __init__(self, file_path, t: str, transform=None):
        hf = h5py.File(file_path, "r")

        self.transform = transform

        if t in ["train", "val"]:
            self.images = hf[t]["images"]
            self.diagnoses = hf[t]["diagnosis"]
        elif t == "test":
            self.images = hf["images"]
            self.diagnoses = hf["diagnosis"]
        else:
            raise ValueError(f"Unknown test/train/val specifier: {t}")

    def __getitem__(self, index):

        image = self.images[index]

        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size)
        image = np.moveaxis(image, -1, 0)

        # select random annotation
        label = np.round(self.diagnoses[index, :].mean())

        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        label = torch.as_tensor(label).long()

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.images.shape[0]
