# Drone_BD_ControlArea

Ding, Ning, et al. ["Estimation of control area in badminton doubles with pose information from top and back view drone videos."](https://arxiv.org/abs/2305.04247) arXiv preprint arXiv:2305.04247 (2023), to appear in Multimedia Tools and Applications.

## Drone Video Download
Top-view and Back-view drone videos can be downloaded from [here](https://drive.google.com/drive/folders/1sIKIDLjyhccO_y6gXeaIkr_1gu1o0vYw?usp=drive_link)

## Dataset Details

| -/-       | Top-View Camera | Back-View Camera |
| --------- | -------- | -------- | 
| Device    | DJI Air 2S   | DJI Air 2S   |
| FPS       | 30           |    30        | 
| Bounding Box     |    |    |
| Shuttlecock      |    |    |




## Usage
- The processed data can be downloaded from [here](https://drive.google.com/file/d/1DcaLrBW0IGFKLnvDKuqXlVz0PWTNU6Pz/view?usp=drive_link)
- Pretrained weights can be downloaded from: 
- For training and testing, please run `python main.py`
- For control area visualization, please run `python visualize.py --checkpoint_path './epo40_lr1e-06_w0_A0.8_B0_G3_K0.03_L4/model_X.pth' 
