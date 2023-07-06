import torch
import numpy as np


def get_dice_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    output = output.view(-1)
    mask = mask.view(-1)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_soft_dice(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average dice of B items
    """
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dice_item_thres = get_dice_threshold(output, mask, thres)
            dice_item_thres_list.append(dice_item_thres.data)
        dice_item_thres_mean = np.mean(dice_item_thres_list)
        dice_list.append(dice_item_thres_mean)

    return np.mean(dice_list)


def get_iou_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)

    intersection = (output * mask).sum()
    total = (output + mask).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def get_soft_iou(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average iou of B items
    """
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            iou_item_thres = get_iou_threshold(output, mask, thres)
            iou_item_thres_list.append(iou_item_thres)
        iou_item_thres_mean = np.mean(iou_item_thres_list)
        iou_list.append(iou_item_thres_mean)

    return np.mean(iou_list)

# =========== GED ============= #


def segmentation_scores(mask1, mask2):
    IoU = get_iou_threshold(mask1, mask2, threshold=0.5)
    return 1.0 - IoU


def generalized_energy_distance(label_list, pred_list):
    label_label_dist = [segmentation_scores(label_1, label_2) for i1, label_1 in enumerate(label_list)
                        for i2, label_2 in enumerate(label_list) if i1 != i2]
    pred_pred_dist = [segmentation_scores(pred_1, pred_2) for i1, pred_1 in enumerate(pred_list)
                      for i2, pred_2 in enumerate(pred_list) if i1 != i2]
    pred_label_list = [segmentation_scores(pred, label) for i, pred in enumerate(pred_list)
                       for j, label in enumerate(label_list)]
    GED = 2 * sum(pred_label_list) / len(pred_label_list) \
          - sum(label_label_dist) / len(label_label_dist) - sum(pred_pred_dist) / len(pred_pred_dist)
    return GED


def get_GED(batch_label_list, batch_pred_list):
    """
    :param batch_label_list: list_list
    :param batch_pred_list:
    :return:
    """
    batch_size = len(batch_pred_list)
    GED = 0.0
    for idx in range(batch_size):
        GED_temp = generalized_energy_distance(label_list=batch_label_list[idx], pred_list=batch_pred_list[idx])
        GED = GED + GED_temp
    return GED / batch_size
