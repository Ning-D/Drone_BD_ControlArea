# Drone_BD_ControlArea

Ding, Ning, et al. ["Estimation of control area in badminton doubles with pose information from top and back view drone videos."](https://arxiv.org/abs/2305.04247) arXiv preprint arXiv:2305.04247 (2023).

## Drone Video Download
Top-view and Back-view drone vidoes can be downloaded from [here](https://www.dropbox.com/scl/fo/0xsa463je2nalimvp1eb6/h?rlkey=lpq0w7l7yrtg2jomi2tzgtdbz&dl=0)
## Processed Data
Processed data can be downloaded from [here](https://www.dropbox.com/s/oov70lliawa1zgv/CA.tar.gz?dl=0)

## Usage
- First, download data from:
- Pretrained weights can be download from: 
- For training and testing, please run `python main.py`
- For control area visualization, please run `python save.py --checkpoint_path ./epo30_lr1e-06_w0_A0.5_B0.5_G3_K0.03/model_X.pth`
