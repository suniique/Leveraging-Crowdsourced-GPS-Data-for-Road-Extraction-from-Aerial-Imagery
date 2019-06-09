import os
import torch
import torch.utils.data as data
from .image_augmentation import *
from .gps_render import GPSDataRender, GPSImageRender

class ImageGPSDataset(data.Dataset):
    def __init__(self, image_list, sat_root="", mask_root="",
                 gps_root="", sat_type="png", mask_type="png", gps_typd="data",
                 feature_embedding="", aug_mode="", randomize=True, aug_sampling_rate=None, aug_precision_rate=None):
        self.image_list = image_list
        self.sat_root = sat_root
        self.mask_root = mask_root
        self.gps_root = gps_root
        self.sat_type = sat_type
        self.mask_type = mask_type
        self.randomize = randomize

        if gps_typd == '':
            self.gps_render = None
        elif gps_typd == 'image':
            self.gps_render = GPSImageRender(gps_root)
        elif gps_typd == 'data':
            self.gps_render = GPSDataRender(gps_root, feature_embedding, aug_mode, aug_sampling_rate, aug_precision_rate)


    def _read_image_and_mask(self, image_id):
        if self.sat_root != "":
            img = cv2.imread(os.path.join(
                self.sat_root, "{0}_sat.{1}").format(image_id, self.sat_type))
        else:
            img = None
        mask = cv2.imread(
            os.path.join(
                self.mask_root,  "{}_mask.png").format(image_id), cv2.IMREAD_GRAYSCALE
        )
        if mask is None: print("[WARN] empty mask: ", image_id)
        return img, mask

    def _render_gps_to_image(self, image_id):
        ix, iy = image_id.split('_')
        gps_image = self.gps_render.render(int(ix), int(iy))
        return gps_image

    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, sat, mask, gps_image, randomize=True):

        if randomize:
            if sat is not None:
                sat = randomHueSaturationValue(sat)
            img = self._concat_images(sat, gps_image)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        else:
            img = self._concat_images(sat, gps_image)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        try:
            img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return img, mask


    def __getitem__(self, index):
        image_id = self.image_list[index]
        if self.gps_render is not None:
            gps_img = self._render_gps_to_image(image_id)
        else:
            gps_img = None
        img, mask = self._read_image_and_mask(image_id)
        img, mask = self._data_augmentation(img, mask, gps_img, self.randomize)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask


    def __len__(self):
        return len(self.image_list)

