import os
from sklearn.model_selection import train_test_split
from .data_loader import ImageGPSDataset

def prepare_Beijing_dataset(args):
    image_list = [x[:-9] for x in os.listdir(args.mask_dir) if x.find('mask.png') != -1]
    train_list, val_list = train_test_split(image_list, test_size=args.val_size, random_state=args.random_seed)
    train_dataset = ImageGPSDataset(train_list, args.sat_dir, args.mask_dir, args.gps_dir,
                                         gps_typd=args.gps_type)
    val_dataset = ImageGPSDataset(val_list, args.sat_dir, args.mask_dir, args.gps_dir,
                                       gps_typd=args.gps_type, randomize=False)
    return train_dataset, val_dataset
