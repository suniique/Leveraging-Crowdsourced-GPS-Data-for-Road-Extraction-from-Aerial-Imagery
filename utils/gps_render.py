import os
import pickle
import pandas as pd
import numpy as np
import cv2


class Render():
    def __init__(self):
        pass

    def render(self, ix, iy):
        pass


class GPSImageRender(Render):
    def __init__(self, gps_root, ext):
        self.gps_root = gps_root
        self.ext = ext

    def render(self, ix, iy):
        mid = "{0}_{1}".format(ix, iy)
        gps = cv2.imread(
            os.path.join(self.gps_root, "{0}_gps.{1}".format(mid, self.ext)),
            cv2.IMREAD_GRAYSCALE,
        )
        if gps is None:
            print(
                "error in gps:",
                os.path.join(
                    self.gps_root, "{0}_gps.{1}".format(mid, self.ext)),
            )
        gps = cv2.resize(gps, (1024, 1024))
        gps = np.expand_dims(gps, axis=2)
        return gps


# TODO: function comments

class GPSDataRender(Render):
    def __init__(self, cache, mode):
        self.mode = mode
        self.cache = cache
        self.length = 1024

    def _read_gps_pickle(self, ix, iy):
        with open(os.path.join(self.cache, f'{iy}_{ix}_gps.pkl'), 'rb') as f:
            patchedGPS = pickle.load(f)

        patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                                (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
        return patchedGPS


    def _sparse_to_dense(self, patchedGPS, length=1024):
        gps = np.zeros((length, length, self.selecting_num), np.uint8)
        ratio = length / 1024.
        y = np.array(patchedGPS['lon'] * ratio, np.int)
        x = np.array(patchedGPS['lat'] * ratio, np.int)
        gps[x, y] = 255
        gps = cv2.dilate(gps, np.ones((3, 3)))
        gps = gps[..., None]
        return gps


    def render(self, ix, iy):
        patchedGPS = self._read_gps_pickle(ix, iy)
        gps = self._sparse_to_dense(patchedGPS, length=self.length)
        gps = cv2.resize(gps, (1024, 1024))
        if gps.ndim == 2:
            gps = gps[..., None]
        return gps
