import torch
import os

from framework import Trainer
from utils.datasets import prepare_Beijing_dataset
from networks.unet import Unet, Unet1DConv
from networks.dlinknet import DinkNet34
from networks.resunet import ResUNet34


def get_model(model_name, use_gps=True):
    input_channels = (4 if use_gps else 3)
    if model_name == 'dlink34':
        model = DinkNet34(num_channels=input_channels)
    elif model_name == 'dlink34_1d':
        model = DinkNet34(num_channels=input_channels, encoder_1dconv=2, decoder_1dconv=2)
    elif model_name == 'resunet34':
        model = ResUNet34(num_channels=input_channels)
    elif model_name == 'resunet34_1d':
        model = ResUNet34(num_channels=input_channels, encoder_1dconv=2)
    elif model_name == 'unet':
        model = Unet(in_channel=input_channels)
    elif model_name == 'unet_1d':
        model = Unet1DConv(in_channel=input_channels)
    return model


def get_dataloader(args):
    train_ds, val_ds = prepare_Beijing_dataset(args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
    return train_dl, val_dl


def train(args):
    net = get_model(args.model, args.gps_dir!='')
    train_dl, val_dl = get_dataloader(args)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer)
    trainer.set_train_dl(train_dl)
    trainer.set_validation_dl(val_dl)
    trainer.set_save_path(WEIGHT_SAVE_DIR)

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='dlink34')
    parser.add_argument('--lr', '-lr', type=float, default=2e-4)
    parser.add_argument('--name', '-n', type=str, default='')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--sat_dir', '-s', type=str, default='')
    parser.add_argument('--mask_dir', '-M', type=str, default='')
    parser.add_argument('--gps_dir', '-g', type=str, default='')
    parser.add_argument('--gps_type', '-t', type=str, default='data')
    parser.add_argument('--weight_save_dir', '-W', type=str, default='./weights')
    parser.add_argument('--val_size', '-T', type=float, default=0.2)
    parser.add_argument('--use_gpu', '-G', type=bool, default=True)
    parser.add_argument('--gpu_ids', '-N', type=str, default='0,1')
    parser.add_argument('--workers', '-w', type=int, default=4)
    parser.add_argument('--epochs', '-e', type=int, default=60)
    parser.add_argument('--random_seed', '-r', type=int, default=12345)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size
    
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, args.model)
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.mkdir(WEIGHT_SAVE_DIR)

    train(args)
    print("Finished")
