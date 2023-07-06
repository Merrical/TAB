# -*- coding: utf-8 -*-
import csv
import PIL.Image
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

MEAN_AND_STD = {'mean_rgb': torch.from_numpy(np.array([0.485, 0.456, 0.406])),
                'std_rgb': torch.from_numpy(np.array([0.229, 0.224, 0.225]))}


def get_data_path_list(root, sub_set_list, rater_num):
    data_path_list = list()
    for sub_set in sub_set_list:
        temp_csv_path = root + '/Glaucoma_multirater_' + sub_set + '.csv'
        reader = csv.reader(open(temp_csv_path))
        lines = list(reader)
        lines = lines[1:]

        for index in range(0, int(len(lines) / 7.0)):
            img_index = (1 + rater_num) * index
            mask_index_list = [(1 + rater_num) * index + i for i in range(1, rater_num + 1)]
            temp_data_path = list()
            temp_data_path.append(lines[img_index][1].replace('\\', '/'))
            for mask_index in mask_index_list:
                temp_data_path.append(lines[mask_index][2].replace('\\', '/'))
            data_path_list.append(temp_data_path)
    return data_path_list


class Disc_Cup(data.Dataset):
    def __init__(self, args, data_path_list):  # img+mask1+mask2+...+mask6 (7 columns list)
        super(Disc_Cup, self).__init__()
        self.root = args.dataroot + '/'
        self.data_path_list = data_path_list
        self.num_rater = args.rater_num
        self.scale_size = (args.img_width, args.img_height)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        temp_path = self.data_path_list[index]
        img_path = self.root + temp_path[0]
        img = transforms.ToTensor()(PIL.Image.open(img_path).convert('RGB').resize(self.scale_size))
        for i in range(img.size(0)):
            img[i] = (img[i]-MEAN_AND_STD['mean_rgb'][i]) / MEAN_AND_STD['std_rgb'][i]

        masks = list()
        for i in range(1, self.num_rater+1):
            mask_path_temp = self.root + temp_path[i]
            mask_temp = np.array(PIL.Image.open(mask_path_temp).convert('L').resize(self.scale_size))
            if mask_temp.max() > 1:
                mask_temp = mask_temp / 255.0

            disc = mask_temp.copy()
            disc[disc != 0] = 1
            cup = mask_temp.copy()
            cup[cup != 1] = 0
            mask_ = np.stack((disc, cup))

            masks.append(mask_)
        return {'image': img, 'mask': masks, 'name': temp_path[0]}
