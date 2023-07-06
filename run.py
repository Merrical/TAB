import torch
from utils.generate import generate_output_folder
from models.bulid import build_model
from loss_func.get_loss import get_loss_func
from dataset.get_dataset import getDataset

from trainer.train_riga_tab import train_riga_tab, test_riga_tab


def train(args):
    log_folder, checkpoint_folder, visualization_folder, metrics_folder = generate_output_folder(args)

    # network
    model = build_model(args)

    # load pretrained params
    if args.pretrained == 1:
        params = torch.load(args.pretrained_dir)
        model_params = params['model']
        model.load_state_dict(model_params)

    # dataset
    train_set, valid_set, test_set = getDataset(args, validate=args.validate)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # loss_func
    loss_func = get_loss_func(args)

    if args.net_arch == "TAB" and args.dataset == "RIGA":
        train_riga_tab(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                       loss_func, train_set, valid_set, test_set)


def test(args):
    log_folder, checkpoint_folder, visualization_folder, metrics_folder = generate_output_folder(args)

    # network
    model = build_model(args)

    # load pretrained params
    params = torch.load(checkpoint_folder + "/amp_checkpoint.pt")
    model_params = params['model']
    model.load_state_dict(model_params)

    # dataset
    _, _, test_set = getDataset(args)

    if args.net_arch == "TAB" and args.dataset == "RIGA":
        test_riga_tab(args, visualization_folder, metrics_folder, model, test_set)
