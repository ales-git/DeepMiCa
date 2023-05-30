from torchvision.io import read_image
from torch.utils import data
import torchvision.transforms.functional as TF
import random
import re


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=False,
                 test=False,
                 nomask=False):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.test = test
        self.nomask = nomask

        if transform is False:
            self.inputs = self.sorted_alphanumeric(self.inputs)
            if nomask is False:
                self.targets = self.sorted_alphanumeric(self.targets)

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        if self.nomask is False:
            target_ID = self.targets[index]

        # Load input and target
        x = read_image(input_ID)
        if self.nomask is False:
            y = read_image(target_ID)

        x = x.unsqueeze(0)
        # Normalize images and masks between 0 and 1
        x = x.float()
        x /= 255.0

        if self.nomask is False:
            y = y.unsqueeze(0)
            y = y.float()
            y /= 255.0

        # Data Augmentation
        class CustomRotationTransform:
            """Rotate by one of the given angles."""

            def __init__(self, angles):
                self.angles = angles

            def __call__(self, x, y):
                angle = random.choice(self.angles)
                return TF.rotate(x, angle), TF.rotate(y, angle)

        if self.transform is True:
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

            rot = CustomRotationTransform(angles=[-90, 0, 90])
            x, y = rot(x, y)

        if self.test == True:
            if self.nomask is False:
                return x, y, input_ID
            else:
                return x, input_ID
        else:
            return x, y
