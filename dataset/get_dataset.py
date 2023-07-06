import torch
from dataset.DiscRegion import get_data_path_list, Disc_Cup


def getDataset(args, validate=False):

    if args.dataset == "RIGA":
        train_set_path_list = get_data_path_list(args.dataroot, ['BinRushed', 'MESSIDOR'], 6)
        test_set_path_list = get_data_path_list(args.dataroot, ['Magrabia'], 6)  # rater_num==6
        train_set = Disc_Cup(args, train_set_path_list)
        test_set = Disc_Cup(args, test_set_path_list)

        if validate:  # True
            train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [int(train_set.__len__()*0.8), train_set.__len__()-int(train_set.__len__()*0.8)])
            return train_dataset, valid_dataset, test_set
        else:  # False
            return train_set, None, test_set
