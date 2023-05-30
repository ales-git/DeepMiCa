import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, dir_path, file_name, dim, transform=None, extract_name=False):
        self.dir_path = dir_path
        self.transform = transform
        self.file = file_name
        self.extract_name = extract_name
        self.dim = dim
        self.img = []
        self.label = []
        if file_name is not None:
            f = open(self.file, "r")
            for line in f:
                line = line.strip()
                path_file, l = line.split(" ")[0], line.split(" ")[1]
                self.label.append(l)
                self.img.append(path_file)
            f.close()
        else:
            self.img = os.listdir(self.dir_path)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):

        if self.file is not None:
            i = read_image(self.img[idx])

        else:
            i = read_image(self.dir_path + self.img[idx])

        if i.shape[0] == 3:
            i = i[0].unsqueeze(0)

        if self.transform:
            i = self.transform(i)

        if self.dim is not None:
            final_transform = transforms.Resize((self.dim, self.dim))

            i = final_transform(i.float())
            i /= 255.0

            i = torch.stack([i, i, i], dim=0)
            i = i.squeeze(1)
        else:
            i = i.float()
            i /= 255.0

        if self.label != []:
            # label
            l = torch.tensor(float(self.label[idx]))

        # img name
        img_name = self.img[idx]

        if self.extract_name == True:
            return i, l, img_name
        else:
            if self.label != []:
                return i, l
            else:
                return i
