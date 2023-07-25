import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset
import h5py


# def collate_fn(batch):
#     x = [item[0] for item in batch]
#     y = [item[1] for item in batch]
#     all_ys = [item[2] for item in batch]
#     return torch.stack(x, 0), torch.stack(y, 0), all_ys


# def create_train_val_loaders(file_path1, file_path2, batch_size):
# dataset_train_Chaksu = Chaksu(file_path=file_path1, t="train")
# dataset_val_Chaksu = Chaksu(file_path=file_path1, t="val")
# dataset_train_RIGA = RIGA(file_path=file_path2, t="train")
# dataset_val_RIGA = RIGA(file_path=file_path2, t="val")

# dataset_train = ConcatDataset((dataset_train_Chaksu, dataset_train_RIGA))
# dataset_val = ConcatDataset((dataset_val_Chaksu, dataset_val_RIGA))

# train_loader = DataLoader(
#     dataset_train,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     drop_last=True,
# )
# validation_loader = DataLoader(
#     dataset_val,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     drop_last=True,
# )

# print(
#     "Number of training/validation patches:",
#     (len(dataset_train), len(dataset_val)),
# )

# return train_loader, validation_loader


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
