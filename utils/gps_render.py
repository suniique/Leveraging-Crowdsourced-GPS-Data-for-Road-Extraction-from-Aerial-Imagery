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
    def __init__(self, cache, feature_embedding="", aug_mode="", aug_sampling_rate=None, aug_precision_rate=None):
        self.features = feature_embedding
        self.aug_mode = aug_mode
        self.cache = cache
        self.length = 1024
        self.aug_sampling_rate = aug_sampling_rate
        self.aug_precision_rate = aug_precision_rate
        print("[INFO] aug_mode:", self.aug_mode)
        print("[INFO] aug_sampling_rate: ", self.aug_sampling_rate)
        print("[INFO] aug_precision_rate: ", self.aug_precision_rate)

    def _read_gps_pickle(self, ix, iy):
        with open(os.path.join(self.cache, f'{iy}_{ix}_gps.pkl'), 'rb') as f:
            patchedGPS = pickle.load(f)

        return patchedGPS


    def _sparse_to_dense(self, patchedGPS, length=1024):
        gps = np.zeros((length, length, 1), np.uint8)
        ratio = length / 1024.
        patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                                (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
        y = np.array(patchedGPS['lon'] * ratio, np.int)
        x = np.array(patchedGPS['lat'] * ratio, np.int)
        gps[x, y] = 255
        gps = cv2.dilate(gps, np.ones((3, 3)))
        gps = gps[..., None]
        return gps


    def _gps_augmentation(self, patchedGPS, aug_mode, length=1024):
        if "sampling" in aug_mode:
            if self.aug_sampling_rate is None:
                sampling_rate = np.random.uniform(0.01, 1.0, None)
            else:
                sampling_rate = self.aug_sampling_rate
            patchedGPS = patchedGPS.sample(frac=sampling_rate)

        if "precision" in aug_mode:
            if self.aug_precision_rate is None:
                precision_rate = np.random.uniform(0.1, 1.0, None)
            else:
                precision_rate = self.aug_precision_rate
            patchedGPS['lat'] = np.floor(np.floor(patchedGPS['lat'] * precision_rate) / precision_rate)
            patchedGPS['lon'] = np.floor(np.floor(patchedGPS['lon'] * precision_rate) / precision_rate)

        if "perturbation" in aug_mode:
            num_records = patchedGPS.shape[0]
            sigma = 10 * np.random.rand()
            patchedGPS['lat'] += (sigma *
                                  np.random.randn(num_records)).astype(np.int)
            patchedGPS['lon'] += (sigma *
                                  np.random.randn(num_records)).astype(np.int)

        if "omission" in aug_mode:
            omission_start_x = length * np.random.rand() - length / 2
            omission_start_y = length * np.random.rand() - length / 2
            omission_end_x = omission_start_x + length / 2
            omission_end_y = omission_start_y + length / 2

            patchedGPS = patchedGPS[~((omission_start_y <= patchedGPS['lat']) & (patchedGPS['lat'] < omission_end_y) &
                                      (omission_start_x <= patchedGPS['lon']) & (patchedGPS['lon'] < omission_end_x))]
        return patchedGPS


    def _feature_embedding(self, patchedGPS, length=1024):
        features = self.features.split('-')
        features_embs = [] #np.zeros((length, length, len(features)), np.uint8)
        if "speed" in features:
            m = self._mean_of_feature_value(patchedGPS, "speed", scale_factor=0.1, length=length)
            features_embs.append(m)
        if "maxspeed" in features:
            m = self._max_of_feature_value(patchedGPS, "speed", scale_factor=0.1, length=length)
            features_embs.append(m)
        if "interval" in features:
            m = self._mean_of_feature_value(patchedGPS, "timeinterval", scale_factor=0.5, length=length)
            features_embs.append(m)
        if "maxinterval" in features:
            m = self._mean_of_feature_value(patchedGPS, "timeinterval", scale_factor=0.5, length=length)
            features_embs.append(m)
        if "heading" in features:
            m = self._aggregate_heading(patchedGPS, length=length)
            features_embs.append(m)
        return np.concatenate(features_embs, 2)


    def _mean_of_feature_value(self, patchedGPS, feature_name, scale_factor, length=1024):
        feature_map = np.zeros((length, length, 1),  np.float)
        count = np.zeros((length, length, 1),  np.float)
        ratio = length / 1024.
        y = np.array(patchedGPS['lon'] * ratio, np.int)
        x = np.array(patchedGPS['lat'] * ratio, np.int)
        # to make 99% speed values in [0..255]
        value = np.array(np.clip(patchedGPS[feature_name] * scale_factor, 0, 255),  np.float)

        def additon(x, y, value):
            if 0 <= x < length and 0 <= y < length:
                feature_map[x, y] += value
                count[x, y] += 1
        average_vfunc = np.vectorize(additon)
        average_vfunc(x, y, value)
        feature_map = np.divide(feature_map, count,
                out=np.zeros_like(feature_map), where=(count!=0))# np.nan_to_num(feature_map / count)
        return feature_map.astype(np.uint8)

    def _max_of_feature_value(self, patchedGPS, feature_name, scale_factor, length=1024):
        feature_map = np.zeros((length, length),  np.float)
        ratio = length / 1024.
        y = np.array(patchedGPS['lon'] * ratio, np.int)
        x = np.array(patchedGPS['lat'] * ratio, np.int)
        # to make 99% speed values in [0..255]
        value = np.array(np.clip(patchedGPS[feature_name] * scale_factor, 0, 255),  np.float)

        def maximize(x, y, value):
            if 0 <= x < length and 0 <= y < length:
                feature_map[x, y] = max(value, feature_map[x, y])
        max_vfunc = np.vectorize(maximize)
        max_vfunc(x, y, value)
        return feature_map[..., None].astype(np.uint8)

    def _aggregate_heading(self, patchedGPS, length=1024):
        feature_map = np.zeros((length, length, 2))
        ratio = length / 1024.
        patchedGPS = patchedGPS[(patchedGPS['dir']
                                 <= 360) & (patchedGPS['dir'] > 0)]
        def degree_to_rad(arr):
            return np.pi * arr / 180
        def aevrage(x, y, sin_value, cos_value):
            if 0 <= x < length and 0 <= y < length:
                if feature_map[x, y].any():
                    feature_map[x, y] = (
                        feature_map[x, y] + np.array([sin_value, cos_value])) / 2
                else:
                    feature_map[x, y] = [sin_value, cos_value]
        aevrage_vfunc = np.vectorize(aevrage)
        y = np.array(patchedGPS['lon'] * ratio, np.int)
        x = np.array(patchedGPS['lat'] * ratio, np.int)
        value = degree_to_rad(np.array(patchedGPS['dir']))
        sin_value = np.sin(value) * 127
        cos_value = np.cos(value) * 127
        aevrage_vfunc(x, y, sin_value, cos_value)
        feature_map += 128
        return feature_map.astype(np.uint8)

    def render(self, ix, iy):
        patchedGPS = self._read_gps_pickle(ix, iy)
        # if "precision" in self.aug_mode:
        #     scaled_length = int(self.length * np.random.uniform(0.1, 1.0, None))
        # else:
        #     scaled_length = 1024
        scaled_length = 1024
        if self.aug_mode != "":
            patchedGPS = self._gps_augmentation(patchedGPS, aug_mode=self.aug_mode, length=scaled_length)
        gps = self._sparse_to_dense(patchedGPS, length=scaled_length)
        if self.features != "":
            gps_features_emb = self._feature_embedding(patchedGPS,
                    length=scaled_length)
            gps = np.concatenate([gps, gps_features_emb], 2)
        gps = cv2.resize(gps, (self.length, self.length))
        if gps.ndim == 2:
            gps = gps[..., None]
        return gps
