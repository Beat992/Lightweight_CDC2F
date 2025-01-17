import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataset import Dataset
from dataset.data_utils import CDDataAugmentation


class CDDataset(Dataset):

    def __init__(self,
                 img_path,
                 split_flag,
                 txt_path=None,
                 img_size=256,
                 to_tensor=True,
                 ):
        self.img_path = img_path
        self.flag = split_flag
        self.img_txt_path = os.path.join(img_path, 'txt', f'{split_flag}.txt') if txt_path is None else txt_path
        self.img_list = np.loadtxt(self.img_txt_path, dtype=str)
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.img_size = img_size
        self.to_tensor = to_tensor
        if self.flag == 'train':
            self.aug = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                # random_color_tf=True,
            )
        else:
            self.aug = CDDataAugmentation(
                img_size=self.img_size
            )

    def get_img_label_path_pairs(self):
        img_label_pair_list = {}
        base_path = self.img_path + f'/{self.flag}'
        for idx, did in enumerate(open(self.img_txt_path)):
            image1_name, image2_name, mask_name = did.strip("\n").split(' ')
            img1_file = os.path.join(base_path, image1_name)
            img2_file = os.path.join(base_path, image2_name)
            lbl_file = os.path.join(base_path, mask_name)
            img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file])

        return img_label_pair_list

    def __getitem__(self, index):
        img1_path, img2_path, label_path = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2 = np.asarray(Image.open(img2_path).convert('RGB'))
        label = np.asarray(Image.open(label_path)).astype(np.float32)

        [img1, img2], [label] = self.aug.transform([img1, img2], [label], to_tensor=self.to_tensor)

        return img1, img2, label

    def __len__(self):

        return len(self.img_label_path_pairs)


def get_data_loader(img_path, split_flag, batch_size,
                    txt_path=None, img_size=256, to_tensor=True,
                    shuffle=True, num_workers=4, pin_memory=True, drop_last=False):
    dataset = CDDataset(img_path, split_flag, txt_path, img_size, to_tensor)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=drop_last)
    return dataloader
