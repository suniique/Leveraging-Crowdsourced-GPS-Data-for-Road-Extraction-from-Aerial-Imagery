import torch
import os

from framework import Trainer
from utils.datasets import prepare_Beijing_dataset
from networks.unet import Unet, UnetMoreLayers
from networks.dlinknet import DinkNet34, LinkNet34
from networks.resunet import ResUnet, ResUnet1DConv
from networks.deeplabv3plus import DeepLabV3Plus


def get_model(model_name, use_gps=True):
    input_channels = input_channel_num
    if model_name == 'dlink34':
        model = DinkNet34(num_channels=input_channels)
    elif model_name == 'dlink34_1d':
        model = DinkNet34(num_channels=input_channels, encoder_1dconv=0,
                          decoder_1dconv=4)
    elif model_name == 'linknet':
        model = LinkNet34(num_channels=input_channels,
                          decoder_1dconv=0)
    elif model_name == 'linknet_1d':
        model = LinkNet34(num_channels=input_channels,
                          decoder_1dconv=4)
    elif model_name == 'linknet_nores':
        model = LinkNet34(num_channels=input_channels,
                          decoder_1dconv=0, using_resnet=False)
    elif model_name == 'linknet_nores_1d':
        model = LinkNet34(num_channels=input_channels,
                          decoder_1dconv=4, using_resnet=False)
    elif model_name == 'resunet':
        model = ResUnet(num_channels=input_channels)
    elif model_name == 'resunet_1d':
        model = ResUnet1DConv(num_channels=input_channels)
    elif model_name == 'unet':
        model = Unet(in_channel=input_channels, conv1d=False)
    elif model_name == 'unet_1d':
        model = Unet(in_channel=input_channels, conv1d=True)
    elif model_name == 'unet_more':
        model = UnetMoreLayers(in_channel=input_channels, conv1d=False)
    elif model_name == 'unet_more_1d':
        model = UnetMoreLayers(in_channel=input_channels, conv1d=True)
    elif model_name == 'deeplabv3+':
        model = DeepLabV3Plus(n_classes=1, num_channels=input_channels)
    return model


def get_dataloader(args):
    train_ds, val_ds, test_ds = prepare_Beijing_dataset(args)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
    return train_dl, val_dl, test_dl


def train(args):
    net = get_model(args.model, args.gps_dir != '')
    train_dl, val_dl, test_dl = get_dataloader(args)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer)
    if args.weight_load_path != '':
        trainer.solver.load_weights(args.weight_load_path)
    trainer.set_train_dl(train_dl)
    trainer.set_validation_dl(val_dl)
    trainer.set_test_dl(test_dl)
    trainer.set_save_path(WEIGHT_SAVE_DIR)

    trainer.fit(epochs=args.epochs)


def sampling_experiment(args):
    from utils.data_loader import ImageGPSDataset
    import numpy as np
    net = get_model(args.model, args.gps_dir != '')
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer)
    if args.weight_load_path != '':
        trainer.solver.load_weights(args.weight_load_path)

    score_list = []
    for sampling_rate in np.arange(0.1, 1.1, 0.1):
        _, _, test_ds = prepare_Beijing_dataset(
            args, aug_sampling_rate=sampling_rate)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
        _, miou = trainer.fit_one_epoch(test_dl, eval=True)
        print(sampling_rate, miou)
        score_list.append(miou[3].item())

    print(score_list)
    with open(os.path.join(os.path.split(args.weight_load_path)[0], "sampling_experiment.txt"), 'w') as f:
        f.write(str(score_list))
        f.close()


def precision_experiment(args):
    from utils.data_loader import ImageGPSDataset
    import numpy as np
    net = get_model(args.model, args.gps_dir != '')
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer)
    if args.weight_load_path != '':
        trainer.solver.load_weights(args.weight_load_path)

    score_list = []
    for sampling_rate in [0.05] + list(np.arange(0.1, 1.1, 0.1)):
        _, _, test_ds = prepare_Beijing_dataset(
            args, aug_precision_rate=sampling_rate)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
        _, miou = trainer.fit_one_epoch(test_dl, eval=True)
        print(sampling_rate, miou)
        score_list.append(miou[3].item())

    print(score_list)
    with open(os.path.join(os.path.split(args.weight_load_path)[0], "precision_experiment.txt"), 'w') as f:
        f.write(str(score_list))
        f.close()


def predict(args):
    import cv2
    import numpy as np

    net = get_model(args.model, args.gps_dir != '')
    _, _, test_ds = prepare_Beijing_dataset(args)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer)
    if args.weight_load_path != '':
        trainer.solver.load_weights(args.weight_load_path)

    predict_dir = os.path.join(os.path.split(
        args.weight_load_path)[0], "prediction")
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)

    for i, data in enumerate(test_ds):
        image = data[0]
        pred = trainer.solver.pred_one_image(image)
        pred = ((pred) * 255.0).astype(np.uint8)
        pred_filename = os.path.join(predict_dir, f"{i}.png")
        cv2.imwrite(pred_filename, pred)
        print("[DONE] predicted image: ", pred_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='dlink34_1d')
    parser.add_argument('--lr', '-lr', type=float, default=2e-4)
    parser.add_argument('--name', '-n', type=str, default='')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--sat_dir', '-s', type=str,
                        default='dataset/Beijing/train_val/sat')
    parser.add_argument('--mask_dir', '-M', type=str,
                        default='dataset/Beijing/train_val/mask')
    parser.add_argument('--test_sat_dir', type=str,
                        default='dataset/Beijing/test/sat')
    parser.add_argument('--test_mask_dir', type=str,
                        default='dataset/Beijing/test/mask')
    parser.add_argument('--gps_dir', '-g', type=str,
                        default='dataset/Beijing/gps_data/')
    parser.add_argument('--gps_type', '-t', type=str, default='data')
    parser.add_argument('--feature_embedding', '-F', type=str, default='')
    parser.add_argument('--gps_augmentation', '-A', type=str, default='')
    parser.add_argument('--test_gps_augmentation', type=str, default='')
    parser.add_argument('--weight_save_dir', '-W',
                        type=str, default='./weights')
    parser.add_argument('--weight_load_path', '-L', type=str, default='')
    parser.add_argument('--val_size', '-T', type=float, default=0.1)
    parser.add_argument('--use_gpu', '-G', type=bool, default=True)
    parser.add_argument('--gpu_ids', '-N', type=str, default='0,1')
    parser.add_argument('--workers', '-w', type=int, default=4)
    parser.add_argument('--epochs', '-e', type=int, default=60)
    parser.add_argument('--random_seed', '-r', type=int, default=12345)
    parser.add_argument('--eval', '-E', type=str, default="")
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size

    if args.sat_dir == "" and args.gps_dir != "":
        input_channels = "gps_only"
        input_channel_num = 1
    elif args.sat_dir != "" and args.gps_dir == "":
        input_channels = "image_only"
        input_channel_num = 3
    elif args.sat_dir != "" and args.gps_dir != "":
        input_channels = "image_gps"
        input_channel_num = 4
    else:
        print("[ERROR] Both input source are empty!")
        exit(1)

    if args.feature_embedding != "":
        num_embedding = args.feature_embedding.split('-')
        input_channel_num += len(num_embedding)
        if "heading" in num_embedding:
            input_channel_num += 1
        print("[INFO] gps embedding: ", num_embedding)

    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                   f"{args.model}_{input_channels}_{args.feature_embedding}_{args.gps_augmentation}")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.mkdir(WEIGHT_SAVE_DIR)

    print("[INFO] input: ", input_channels)
    print("[INFO] channels: ", input_channel_num)

    if args.eval == "":
        train(args)
        print("[DONE] training finished")
    elif args.eval == "sampling_experiment":
        sampling_experiment(args)
        print("[DONE] sampling finished")
    elif args.eval == "precision_experiment":
        precision_experiment(args)
        print("[DONE] precision_experiment finished")
    elif args.eval == "predict":
        predict(args)
        print("[DONE] predict finished")
