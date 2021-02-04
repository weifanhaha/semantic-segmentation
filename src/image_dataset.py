#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms as T
from PIL import Image
import glob


# In[2]:


class ImageDataset(Dataset):
    def __init__(self, mode, predict_img_path=''):
        assert mode in ["train", "val", "test", "predict"]

        self.mode = mode
        self.init_img_paths(mode, predict_img_path)
        self.init_transforms()

        self.len = len(self.img_paths)

    def init_img_paths(self, mode, predict_img_path):
        train_img_path = '../satellite_images/p2_data/train/'
        test_img_path = '../satellite_images/p2_data/validation/'

        train_sat_img_paths = sorted(glob.glob(train_img_path + '*.jpg'))
        train_mask_img_paths = sorted(glob.glob(train_img_path + '*.png'))
        test_sat_img_paths = sorted(glob.glob(test_img_path + '*.jpg'))
        test_mask_img_paths = sorted(glob.glob(test_img_path + '*.png'))
        predict_img_paths = sorted(glob.glob(predict_img_path + '*.jpg'))

        if mode == 'train':
            self.img_paths = train_sat_img_paths[:1800]
        elif mode == 'val':
            self.img_paths = train_sat_img_paths[1800:]
        elif mode == 'test':
            self.img_paths = test_sat_img_paths
        else:
            self.img_paths = predict_img_paths

    def init_transforms(self):
        self.transforms = T.ToTensor()

    # mask to 7 channel ?
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split('/')[-1]
        img_id = img_name[:-8]

        # get image tensor
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transforms(image)

        # get mask image tensor
        if self.mode == 'predict':
            return (image_tensor, img_id)

        mask_img_path = img_path[:-8] + '_mask.png'
        mask_image = Image.open(mask_img_path).convert("RGB")
        mask_image_array = np.array(mask_image) / 255
        label_image_array = self.mask2label(mask_image_array)
        label_image_tensor = torch.tensor(label_image_array, dtype=torch.long)
#         label_image_tensor = torch.tensor(label_image_array)

#         mask_image_tensor = self.transforms(mask_image)
        return (image_tensor, label_image_tensor)

    def __len__(self):
        return self.len

    def mask2label(self, img_array):
        img = 4 * img_array[:, :, 0] + 2 * \
            img_array[:, :, 1] + img_array[:, :, 2]
        label_matrix = np.zeros((512, 512))

        label_matrix[np.where(img == 3)] = 0  # (Cyan: 011) Urban land
        label_matrix[np.where(img == 6)] = 1  # (Yellow: 110) Agriculture land
        label_matrix[np.where(img == 5)] = 2  # (Purple: 101) Rangeland
        label_matrix[np.where(img == 2)] = 3  # (Green: 010) Forest land
        label_matrix[np.where(img == 1)] = 4  # (Blue: 001) Water
        label_matrix[np.where(img == 7)] = 5  # (White: 111) Barren land
        label_matrix[np.where(img == 0)] = 6  # (Black: 000) Unknown
        label_matrix[np.where(img == 4)] = 6  # (Red: 100) Unknown

        return label_matrix


# In[165]:


# train_dataset = ImageDataset("train")
# train_dataloader = DataLoader(train_dataset, batch_size=32)


# In[166]:


# image_tensor, label_tensor = next(iter(train_dataloader))
