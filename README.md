# Drone_BD_ControlArea

Ding, Ning, et al. ["Estimation of control area in badminton doubles with pose information from top and back view drone videos."](https://doi.org/10.1007/s11042-023-16362-1) *Multimedia Tools and Applications* (2023). 
## Drone Video Download
Top-view and Back-view drone videos can be downloaded from [here](https://drive.google.com/drive/folders/1sIKIDLjyhccO_y6gXeaIkr_1gu1o0vYw?usp=drive_link)

## Dataset Details

| -/-       | Top-View Camera | Back-View Camera |
| --------- | -------- | -------- | 
| Device    | DJI Air 2S   | DJI Air 2S   |
| FPS       | 30           |    30        | 
| Bounding Box     |  ✓  |  ✓  |
| Shuttlecock (hit/drop)     |  ✓  |  ✓  |




## Usage
- The processed data can be downloaded from [here](https://drive.google.com/file/d/1DcaLrBW0IGFKLnvDKuqXlVz0PWTNU6Pz/view?usp=drive_link)
- Pretrained weights can be downloaded from: 
- For training and testing, please run `python main.py`, parameters can be modified in configuration.py
- For control area visualization, please run `python visualize.py --checkpoint_path`
