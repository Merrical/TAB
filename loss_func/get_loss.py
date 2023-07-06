import torch.nn as nn


def get_loss_func(args):
    if args.loss_func == "bce":
        loss_func = nn.BCEWithLogitsLoss()
    elif args.loss_func == "ce":
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.CrossEntropyLoss()
    return loss_func
