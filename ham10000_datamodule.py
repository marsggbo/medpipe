
from typing import List, Optional, Tuple
import torch
import unittest
import numpy as np
import torchvision.transforms as transforms

from datasets import load_from_disk
from torchvision.transforms import v2
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class Ham10000Dataset(Dataset):
    # default_train_transforms = v2.Compose([
    #     v2.PILToTensor(),
    #     v2.RandomResizedCrop(size=(448, 608), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.RandomVerticalFlip(p=0.5),
    #     v2.RandomRotation(degrees=45),
    #     v2.ToDtype(torch.float32),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # default_valid_transforms = v2.Compose([
    #     v2.PILToTensor(),
    #     v2.Resize(size=(448, 608)),
    #     v2.ToDtype(torch.float32),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # Define default train transforms
    default_train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(448, 608), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define default valid transforms
    default_valid_transforms = transforms.Compose([
        transforms.Resize(size=(448, 608)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, dataset_dir, transform=None, add_random_transforms: bool=False):
        self.dataset = load_from_disk(dataset_dir)
        labels = sorted(list(set(self.dataset['dx'])))
        self.class2idx = {
            labels[i]: i for i in range(len(labels))
        }
        if transform is not None:
            self.transform = transform
        else:
            if not add_random_transforms:
                self.transform = self.default_valid_transforms
            else:
                self.transform = self.default_train_transforms
            
        assert self.class2idx == {
            'actinic_keratoses': 0,
            'basal_cell_carcinoma': 1,
            'benign_keratosis-like_lesions': 2,
            'dermatofibroma': 3,
            'melanocytic_Nevi': 4,
            'melanoma': 5,
            'vascular_lesions': 6
        }, f"Unexpected class2idx mapping: {self.class2idx}"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert idx <= len(self.dataset), f'idx {idx} is out of range for dataset {len(self.dataset)}'
        example = self.dataset[idx]
        image = example['image']
        label = self.class2idx[example['dx']]
        if self.transform:
            image = self.transform(image) # 3*448*608
        return image, label


class Ham10000DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir="/data2/share/skin_cancer_data/",
        concat_train_val: bool=True, # whether to concat train and valid dataset
        batch_size: int=16,
        add_random_transforms: bool=False,
        num_workers: int = 4,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.add_random_transforms = add_random_transforms
        self.train_dataset = Ham10000Dataset(f"{root_dir}/skin_cancer_train", add_random_transforms=add_random_transforms)
        self.valid_dataset = Ham10000Dataset(f"{root_dir}/skin_cancer_val")
        self.test_dataset = Ham10000Dataset(f"{root_dir}/skin_cancer_test")
        if concat_train_val:
            self.train_dataset = ConcatDataset([self.train_dataset, self.valid_dataset])
            self.valid_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# for test only
class TestHam10000DataModule(unittest.TestCase):
    def test_dataset_class(self):
        dataset_dir = "/data2/share/skin_cancer_data/skin_cancer_train"
        dataset = Ham10000Dataset(dataset_dir)
        example = dataset[0]
        self.assertIsInstance(dataset, Dataset)

    def test_data_module(self):
        # 测试数据模块的基本功能
        data_module = Ham10000DataModule(root_dir="/data2/share/skin_cancer_data/")
        self.assertIsInstance(data_module, LightningDataModule)

        # 测试训练数据加载器
        train_loader = data_module.train_dataloader()
        self.assertIsInstance(train_loader, DataLoader)

        # 测试验证数据加载器
        valid_loader = data_module.val_dataloader()
        self.assertIsInstance(valid_loader, DataLoader)

        # 测试测试数据加载器
        test_loader = data_module.test_dataloader()
        self.assertIsInstance(test_loader, DataLoader)

        for idx, data in enumerate(test_loader):
            if idx == 10:
                break
            print(f"batch{idx} {data[0].shape}")

if __name__ == '__main__':
    # unittest.main()
    dataset_dir = "/data2/share/skin_cancer_data/skin_cancer_train"
    dataset = Ham10000Dataset(dataset_dir)
    example = dataset[0]