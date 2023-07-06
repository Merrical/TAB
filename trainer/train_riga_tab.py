import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.functions import adjust_learning_rate
from utils.metrics import get_soft_dice, get_GED
from torch.distributions import Normal, Independent


def train_riga_tab(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder,
                   model, optimizer, loss_func, train_set, valid_set, test_set):
    lambda_ = args.lambda_
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.cuda()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    writer = SummaryWriter(log_dir=log_folder)
    amp_grad_scaler = GradScaler()

    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', (n_trainable_parameters / 1.0e+6), 'M')

    for this_epoch in range(args.num_epoch):
        print(this_epoch)
        model.train()
        train_loss = 0.0
        train_soft_dice_disc = 0.0
        train_soft_dice_cup = 0.0

        for step, data in enumerate(train_loader):
            imgs = data['image'].to(dtype=torch.float32).cuda()
            mask = data['mask']

            optimizer.zero_grad()
            with autocast():
                mask_major_vote = torch.stack(mask, dim=0).sum(dim=0) / args.rater_num
                mask_major_vote = mask_major_vote.to(dtype=torch.float32).cuda()
                global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples = model(imgs, training=True)

                loss_global, loss_rater = 0.0, 0.0
                for i in range(args.rater_num):
                    rater_mask = mask[i].cuda()
                    loss_global += loss_func(global_samples[:, i], rater_mask)
                    loss_rater += loss_func(rater_samples[:, i], rater_mask)
                loss = loss_global + lambda_ * loss_rater

            adjust_learning_rate(optimizer, this_epoch, args.learning_rate, args.num_epoch, args.power)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            train_loss = train_loss + loss.item() * imgs.size(0)
            global_mu = torch.sigmoid(global_mu).to(dtype=torch.float32)
            train_soft_dice_cup = train_soft_dice_cup + get_soft_dice(outputs=global_mu[:, 1, :, :].cpu(),
                                                                      masks=mask_major_vote[:, 1, :, :].cpu()) * imgs.size(0)
            train_soft_dice_disc = train_soft_dice_disc + get_soft_dice(outputs=global_mu[:, 0, :, :].cpu(),
                                                                        masks=mask_major_vote[:, 0, :, :].cpu()) * imgs.size(0)
        writer.add_scalar("Loss/train", train_loss / train_set.__len__(), this_epoch)
        writer.add_scalar("Train/voting_disc", train_soft_dice_disc / train_set.__len__(), this_epoch)
        writer.add_scalar("Train/voting_cup", train_soft_dice_cup / train_set.__len__(), this_epoch)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_folder + '/amp_checkpoint.pt')

    if args.validate:
        test_riga_tab(args, visualization_folder, metrics_folder, model, valid_set)
    else:
        test_riga_tab(args, visualization_folder, metrics_folder, model, test_set)


def test_visualization(imgs_name, Preds_visual, visualization_folder):
    no_samples = Preds_visual.size(0)
    for idx in range(no_samples):
        Pred = np.uint8(Preds_visual[idx].detach().cpu().numpy() * 255)
        Pred_disc = cv2.applyColorMap(Pred[0], cv2.COLORMAP_JET)
        Pred_cup = cv2.applyColorMap(Pred[1], cv2.COLORMAP_JET)
        Pred_path = visualization_folder + '/' + imgs_name[idx][15:-4] + '_Pred_TAB.png'
        cv2.imwrite(Pred_path, 0.5 * Pred_disc + 0.5 * Pred_cup)


def test_riga_tab(args, visualization_folder, metrics_folder, model, test_set):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    metrix_file = metrics_folder + "/dice.txt"
    file_handle = open(metrix_file, 'a')
    if args.validate:
        file_handle.write('validation data size: %d \n' % (test_set.__len__()))
    else:
        file_handle.write('testing data size: %d \n' % (test_set.__len__()))
    file_handle.close()

    model.eval()
    test_soft_dice_cup = 0.0
    test_soft_dice_disc = 0.0

    test_soft_dice_disc_raters = [0.0] * args.rater_num
    test_soft_dice_cup_raters = [0.0] * args.rater_num

    test_GED_disc = 0.0
    test_GED_cup = 0.0

    imgs_visual = list()
    Preds_visual = list()

    for step, data in enumerate(test_loader):
        with torch.no_grad():
            imgs = data['image'].to(dtype=torch.float32).cuda()
            mask = data['mask']

            mask_major_vote = torch.stack(mask, dim=0).sum(dim=0) / args.rater_num
            mask_major_vote = mask_major_vote.to(dtype=torch.float32)
            global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples = model(imgs, training=False)

            rater_mus_sigmoid = torch.sigmoid(rater_mus)
            global_mu_sigmoid = torch.sigmoid(global_mu)

            imgs_visual.extend(data['name'])
            Preds_visual.append(global_mu_sigmoid)

            test_soft_dice_cup = test_soft_dice_cup + get_soft_dice(outputs=global_mu_sigmoid[:, 1, :, :].cpu(),
                                                                    masks=mask_major_vote[:, 1, :, :].cpu()) * imgs.size(0)
            test_soft_dice_disc = test_soft_dice_disc + get_soft_dice(outputs=global_mu_sigmoid[:, 0, :, :].cpu(),
                                                                      masks=mask_major_vote[:, 0, :, :].cpu()) * imgs.size(0)

            test_soft_dice_disc_raters = [
                test_soft_dice_disc_raters[i] + get_soft_dice(outputs=rater_mus_sigmoid[:, i][:, 0, :, :].cpu(),
                                                              masks=mask[i][:, 0, :, :].cpu()) * imgs.size(0) for i in range(args.rater_num)]
            test_soft_dice_cup_raters = [
                test_soft_dice_cup_raters[i] + get_soft_dice(outputs=rater_mus_sigmoid[:, i][:, 1, :, :].cpu(),
                                                             masks=mask[i][:, 1, :, :].cpu()) * imgs.size(0) for i in range(args.rater_num)]

            global_dist = Independent(Normal(loc=global_mu, scale=global_sigma, validate_args=False), 1)
            output_sample_list = []
            for idx_sample in range(20):
                output_sample = global_dist.sample()
                output_sample_list.append(output_sample.cpu())

            cup_label_list, disc_label_list, cup_pred_list, disc_pred_list = get_label_pred_list(args, imgs.size(0), mask, output_sample_list)
            test_GED_disc = test_GED_disc + get_GED(batch_label_list=disc_label_list,
                                                    batch_pred_list=disc_pred_list) * imgs.size(0)
            test_GED_cup = test_GED_cup + get_GED(batch_label_list=cup_label_list,
                                                  batch_pred_list=cup_pred_list) * imgs.size(0)

    Preds_visual = torch.cat(Preds_visual, dim=0)
    test_visualization(imgs_visual, Preds_visual, visualization_folder)

    file_handle = open(metrix_file, 'a')
    file_handle.write("Mean Voting: ({}, {})\n".format(round(test_soft_dice_disc / test_set.__len__() * 100, 2),
                                                       round(test_soft_dice_cup / test_set.__len__() * 100, 2)))
    file_handle.write(
        "Average: ({}, {})\n".format(round(np.mean(test_soft_dice_disc_raters) / test_set.__len__() * 100, 2),
                                     round(np.mean(test_soft_dice_cup_raters) / test_set.__len__() * 100, 2)))

    for i in range(args.rater_num):
        file_handle.write(
            "rater{}: ({}, {})\n".format(i + 1, round(test_soft_dice_disc_raters[i] / test_set.__len__() * 100, 2),
                                         round(test_soft_dice_cup_raters[i] / test_set.__len__() * 100, 2)))

    file_handle.write("GED: ({}, {})\n".format(round(test_GED_disc.item() / test_set.__len__(), 4),
                                               round(test_GED_cup.item() / test_set.__len__(), 4)))


def get_label_pred_list(args, bs, masks, output_sample_list):
    cup_label_list = []
    disc_label_list = []
    for idx in range(bs):
        temp_cup_label_list = []
        temp_disc_label_list = []
        for anno_no in range(args.rater_num):
            temp_cup_label = masks[anno_no][idx, 1, :, :].to(dtype=torch.float32)
            temp_disc_label = masks[anno_no][idx, 0, :, :].to(dtype=torch.float32)
            temp_cup_label_list.append(temp_cup_label)
            temp_disc_label_list.append(temp_disc_label)
        cup_label_list.append(temp_cup_label_list)
        disc_label_list.append(temp_disc_label_list)

    cup_pred_list = []
    disc_pred_list = []
    for idx in range(bs):
        temp_cup_pred_list = []
        temp_disc_pred_list = []
        for pred_no in range(len(output_sample_list)):
            temp_cup_pred = torch.sigmoid(output_sample_list[pred_no][idx, 1, :, :])
            temp_disc_pred = torch.sigmoid(output_sample_list[pred_no][idx, 0, :, :])
            temp_cup_pred_list.append(temp_cup_pred)
            temp_disc_pred_list.append(temp_disc_pred)
        cup_pred_list.append(temp_cup_pred_list)
        disc_pred_list.append(temp_disc_pred_list)
    return cup_label_list, disc_label_list, cup_pred_list, disc_pred_list
