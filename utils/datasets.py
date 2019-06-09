import os
from sklearn.model_selection import train_test_split
from .data_loader import ImageGPSDataset

def prepare_Beijing_dataset(args, aug_sampling_rate=None, aug_precision_rate=None):
    image_list = [x[:-9] for x in os.listdir(args.mask_dir) if x.find('mask.png') != -1]
    test_list = [x[:-9] for x in os.listdir(args.test_mask_dir) if x.find('mask.png') != -1]
    train_list, val_list = train_test_split(image_list, test_size=args.val_size, random_state=args.random_seed)
    train_dataset = ImageGPSDataset(train_list, args.sat_dir, args.mask_dir, args.gps_dir,
                                         gps_typd=args.gps_type,
                                         feature_embedding=args.feature_embedding, aug_mode=args.gps_augmentation)
    val_dataset = ImageGPSDataset(val_list, args.sat_dir, args.mask_dir, args.gps_dir,
                                       gps_typd=args.gps_type, feature_embedding=args.feature_embedding, randomize=False)
    test_dataset = ImageGPSDataset(test_list, args.test_sat_dir, args.test_mask_dir, args.gps_dir,
                                   gps_typd=args.gps_type,
                                   feature_embedding=args.feature_embedding, randomize=False,
                                   aug_mode=args.test_gps_augmentation,
                                   aug_sampling_rate=aug_sampling_rate, aug_precision_rate=aug_precision_rate)

    return train_dataset, val_dataset, test_dataset
