import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PetfinderPawpularityDataset(Dataset):
    def __init__(self, data_frame, transform=None, target_transform=None):

        # storage these params for use in getitem
        self.data_frame = data_frame
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame["Id"].iloc[idx]

        image = Image.open(self.data_frame["Path"].iloc[idx])
        if type(self.transform) == transforms.Compose:
            image = self.transform(image)
        else:
            image = self.transform(image=np.array(image))["image"]
        label = (
            self.data_frame["Pawpularity"].iloc[idx]
            if "Pawpularity" in self.data_frame.columns
            else image_id
        )
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
