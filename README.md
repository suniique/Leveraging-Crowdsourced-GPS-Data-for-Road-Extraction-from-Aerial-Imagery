# Leveraging Crowdsourced GPS Data for Road Extraction from Aerial Imagery
The source code of CVPR 2019 paper *"Leveraging Crowdsourced GPS Data for Road Extraction from Aerial Imagery"*

## Usage

```bash
python train.py \
	--model "dlink34" \
	--sat_dir "dataset/train_val/image" \
	--mask_dir "dataset/train_val/mask"\
	--gps_dir "dataset/GPS/patch" \
	--gps_type "data"
	
```


## Dataset

Our dataset is avaliable upon request. Please contact via: suntao [AT] tongji.edu.cn

### Dataset Description

- **train_val/**
  - **image/**: contains 278 satellite images (`x_y_sat.png `)
  - **mask/**: contains 278 mask images (`x_y_mask.png `)
- **test/**
  - **image/**: contains 70 satellite images (`x_y_sat.png `)
  - **mask/**: contains 70 mask images (`x_y_mask.png `)
- **GPS/**
  - `beijing_gps_dir_speed_interval_sorted.pkl`: The pickle file storing all raw GPS records
  - **patch/**: contains 348 GPS patch files (`x_y_gps.pkl`). Each stores the GPS records located in the area of input image `x_y_sat.png`
- **coordinate/**: contains `x_y_coor.txt` (WGS format) and `x_y_coor2.txt` (GCJ format) files

Each input image `image/x_y_sat.png ` is a RGB satellite image of 1024 $\times$ 1024 pixel size. Its corresponding GPS data is stored in file  `/GPS/patch/x_y_gps.pkl`, and corresponding mask image is   `mask/x_y_mask.png`.

Unfortunately, we haven't got the permission to publish the satellite images due to the license of the data provider. However, we provide all the GPS coordinates of each satellite image (avaliable in WSG and GCJ format) in `/coordinate/`. You might apply for the access and download these images from Amap (高德地图) or DigitalGlobe referencing the coordinates.

### GPS Data

The GPS dataset contains ~50m rows of GPS record collected from ~28k vehicles in Beijing.

To save the loading time, we publish the dataset in Python's Pickle format, which can be directly loaded like:

```python
import pandas
import pickle
gps_data = pickle.load(open('beijing_gps_dir_speed_interval_sorted.pkl', 'rb'))
```

Here are first lines of this file:

|   |ID | time |        lat |      lon |       dir | speed | timeinterval |
| ---: | ---: | ---------: | -------: | --------: | ----: | -----------: | ----- |
|    0 |    0 | 1228061046 | 39.71743 | 116.61815 |     0 |            0 | NaN   |
|    1 |    0 | 1228088457 | 39.71742 | 116.61798 |     0 |            0 | 177.5 |
|    2 |    0 | 1228088520 | 39.71670 | 116.61420 |   159 |            0 | 150.5 |
|    3 |    0 | 1228088758 | 39.71742 | 116.61798 |     0 |            0 | 272.5 |
|    4 |    0 | 1228090926 | 39.71670 | 116.61428 |     0 |            0 | 354.5 |
|    5 |    0 | 1228091249 | 39.73902 | 116.60902 |    12 |          308 | 318.0 |
|    6 |    0 | 1228091562 | 39.73770 | 116.56821 |   267 |         1080 | 264.0 |

**Definition of columns**:

- `ID`: Vehical ID (integer)
- `time`: Timestamp (UNIX format, in second)
- `lat`: Latitude (in degree)
- `lon`: Lontitude (in degree)
- `dir`: Heading (in degree, 0 means the vehical is heading north or isn't moving)
- `speed`: Speed (in meter per minute)
- `timeinterval`: The time interval between two records (in second)

The `lat`/`lon` are in the WGS System. The data table is sorted by `ID`  and then by `time`. 


### License

![img](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)

This dataset is published under [**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike) License . Please note that it can be **ONLY** used for academic or scientific purpose. 

## Citation
We kindly remind you that if found the code or dataset is useful for your research, please cite our paper.
```
Sun, Tao, Zonglin Di, Pengyu Che, Chun Liu, and Yin Wang. 
"Leveraging Crowdsourced GPS Data for Road Extraction From Aerial Imagery" 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7509-7518. 2019.
```
