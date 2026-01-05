import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class cifar10_dataset_labeled(Dataset):
    def __init__(
        self,
        img_file,
        label_file,
        label_size,
        train=True,
        train_transforms=None,
        test_transforms=None,
    ):
        super().__init__()

        self.img_file = np.load(img_file)
        self.label_file = np.load(label_file)

        self.label_size = label_size
        self.train = train
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def __len__(self):
        return self.label_size if self.train else len(self.img_file)

    def __getitem__(self, idx):
        label = self.label_file[idx]
        img = self.img_file[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(img)

        if self.train:
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)

        return image, label, idx


class cifar10_dataset_unlabeled(Dataset):
    def __init__(
        self,
        img_file,
        label_file,
        label_size,
        train_transforms=None,
        limit=None,
    ):
        super().__init__()

        self.img_file = np.load(img_file)
        self.label_file = np.load(label_file)
        self.train_transforms = train_transforms
        self.label_size = label_size

        if limit is not None:
            self.img_file = self.img_file[:limit]
            self.label_file = self.label_file[:limit]

        self.length = len(self.img_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.label_file[idx]
        img = self.img_file[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(img)
        image = self.train_transforms(image)

        # offset index so labeled & unlabeled don't collide
        return image, label, idx + self.label_size
