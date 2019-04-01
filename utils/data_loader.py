import os
import torch
import torch.utils.data as data
from .image_augmentation import *
from .gps_render import GPSDataRender, GPSImageRender

class ImageGPSDataset(data.Dataset):
    def __init__(self, image_list, sat_root="", mask_root="",
                 gps_root="", sat_type="png", mask_type="png", gps_typd="data", gps_render_mode="", randomize=True):
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
            self.gps_render = GPSDataRender(gps_root, gps_render_mode)


    def _read_image_and_mask(self, image_id):
        img = cv2.imread(os.path.join(
            self.sat_root, "{0}_sat.{1}").format(image_id, self.sat_type))
        mask = cv2.imread(
            os.path.join(
                self.mask_root,  "{}_mask.png").format(image_id), cv2.IMREAD_GRAYSCALE
        )
        if img is None: print("[WARN] empty image: ", image_id)
        if mask is None: print("[WARN] empty mask: ", image_id)
        return img, mask

    def _render_gps_to_image(self, image_id):
        ix, iy = image_id.split('_')
        gps_image = self.gps_render.render(int(ix), int(iy))
        return gps_image

    def _data_augmentation(self, img, mask, gps_image, randomize=True):
        if randomize:
            img = randomHueSaturationValue(img)
            if gps_image is not None:
                img = np.concatenate([img, gps_image], 2)
            img, mask = randomShiftScaleRotate(img,mask,)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        else:
            if gps_image is not None:
                img = np.concatenate([img, gps_image], 2)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
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

