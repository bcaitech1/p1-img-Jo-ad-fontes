import random
import numpy as np
import torch
import torch.utils.data as data


from PIL import Image
from glob import glob
from torch.utils.data import Dataset

from albumentations import *
from albumentations.pytorch import ToTensorV2
from RandAugment import RandAugment


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


class eval_data(Dataset):
    def __init__(self, root="./input/data/eval/images", transform=None):
        super(eval_data, self).__init__()
        self.root = root
        self.images = sorted(glob(root + "/*"))
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(image_name)
        img_w, img_h = image.size
        image_transform = self.transform(image=np.array(image))["image"]

        return image_name, image_transform

    def __len__(self):
        return len(self.images)


def get_eval_loader():
    transform = get_transforms()

    eval_dataset = eval_data(transform=transform["val"])

    eval_loader = data.DataLoader(
        eval_dataset, batch_size=90, num_workers=2, shuffle=False
    )

    return eval_loader


class Img_data(Dataset):
    def __init__(self, root="/opt/ml/input/data/train/images", transform=None):
        super(Img_data, self).__init__()
        self.root = root
        self.transform = transform
        self.images = sorted(glob(root + "/**/*"))

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(image_name)
        img_w, img_h = image.size
        label = self.label_attach(image_name)

        # albumentations을 사용하기 위한 형식
        image_transform = self.transform(image=np.array(image))["image"]

        return image_transform, label

    def __len__(self):
        return len(self.images)

    def label_attach(self, info):
        label = None

        arr = info.split("/")[-2:]
        attr = arr[0].split("_")

        img_id, gender, area, age = attr
        mask = arr[1]
        age = int(age)

        # label 0~17 쉽게 하는 방법이 있을 것 같긴한데...
        mask_labeling = {"m": 0, "i": 1, "n": 2}
        gender_labeling = {"male": 0, "female": 1}
        age_labeling = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

        mask_label = mask_labeling[mask[0]]
        gender_label = gender_labeling[gender]
        age_label = age_labeling(age)

        label = self.encode_multi_class(mask_label, gender_label, age_label)

        return label

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label


def get_loader(BATCH_SIZE):
    data_set = Img_data()

    indices = [i for i in range(0, len(data_set))]

    train_data = data.Subset(data_set, indices[:15120])
    val_data = data.Subset(data_set, indices[15120:])

    transform = get_transforms()

    train_data.dataset.set_transform(transform["train"])
    val_data.dataset.set_transform(transform["val"])

    train_loader = data.DataLoader(
        train_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, pin_memory=True
    )

    val_loader = data.DataLoader(
        val_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader


def get_transforms(
    need=("train", "val"),
    img_size=(384, 384),
    mean=(0.5601, 0.5241, 0.5015),
    std=(0.233, 0.243, 0.24567),
):
    transformations = {}
    if "train" in need:
        transformations["train"] = Compose(
            [
                Resize(img_size[0], img_size[1], p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    if "val" in need:
        transformations["val"] = Compose(
            [
                Resize(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    return transformations


if __name__ == "__main__":
    pass
