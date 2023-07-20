import argparse
import numpy as np
from run import train, test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--dataroot", type=str, default='/.../datasets/DiscRegion')
    parser.add_argument("--dataset", choices=["RIGA"], default="RIGA")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--net_arch", choices=["TAB"], default="TAB")
    parser.add_argument("--rater_num", type=int, default=6)
    parser.add_argument("--loss_func", choices=["bce"], default="bce")

    # pretrained params
    parser.add_argument("--pretrained", type=int, default=0, help="whether to load pretrained models.")
    parser.add_argument("--pretrained_dir", type=str, default="none", help="the path of pretrained models.")

    # details of dataset
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_channel", type=int, default=3)

    # training settings: classes; bs; lr; EPOCH; device_id
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--power", default=0.9, type=float, help="poly")
    parser.add_argument("--num_epoch",  default=300, type=int)
    parser.add_argument("--device_id", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], help="gpu ID.")
    parser.add_argument("--loop", default=0, type=int, help="this is the {loop}-th run.")

    parser.add_argument("--rank", default=8, type=int)

    # hyper-parameter of loss
    parser.add_argument("--lambda_", default=1.25, type=float)

    # TAB parameters
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    #  # * CNN encoder
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet34', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=6, type=int)  # 4
    #  # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=1, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=7, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    #  # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    #  # * Segmentation
    parser.add_argument('--masks', action='store_true')

    args = parser.parse_args()
    print(args)

    if args.phase == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
