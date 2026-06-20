import torch

from torch.utils.data import Dataset
from PIL import Image


# to use PyTorch training 
# we have to store our data in its Dataset concept
# only requires to have __len__() and __getitem__() - rest is optional
# make sure to have transform (normalizing data, and other transformations - if needed)
#     and return PyTorch-specific formats - e.g., tensors for data/labels, etc.
class ImageData(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])

        # apply transformation so it's ready for pytorch training
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        label = torch.tensor(label) # convert to tensor as this is passed to pytorch library for training

        return img, label
    
    def open_img(self, idx):
        img = Image.open(self.file_list[idx])
        return img