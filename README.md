# Drone_BD_ControlArea--Drone Video Dataset for Badminton Analysis

This repository contains a unique dataset of top-view and back-view drone videos capturing badminton matches. Accompanying these videos are CSV files containing data for bounding boxes around players and the shuttlecock's position. This dataset is ideal for projects involving sports analytics, motion tracking, and computer vision applications.

Ding, Ning, et al. ["Estimation of control area in badminton doubles with pose information from top and back view drone videos."](https://doi.org/10.1007/s11042-023-16362-1) *Multimedia Tools and Applications* (2023). 

<div align="center">
  <img src="https://github.com/Ning-D/Drone_BD_ControlArea/blob/main/visual/Rally.gif" alt="Description of the second GIF" width="300">
  <img src="https://github.com/Ning-D/Drone_BD_ControlArea/blob/main/visual/Estimation.gif" alt="Estimation of control area in a rally" width="400">
  
</div>






## Dataset Download

Access the drone video dataset and corresponding CSV files via this Google Drive link:

[Download Drone Video Dataset](https://drive.google.com/drive/folders/1sIKIDLjyhccO_y6gXeaIkr_1gu1o0vYw?usp=drive_link)

### Contents

The dataset includes:

- Top-view drone videos
- Back-view drone videos
- CSV files with bounding box coordinates for 4 players and the shuttlecock

## Visualization Tool

A Python script named `plot.py` is provided to overlay bounding boxes and the shuttle position onto the videos. Follow these steps to use the script:

1. **Prepare the Data**:
   - Download the dataset using the provided link.
   - Place the videos and CSV files in the designated directories (as structured in the repository).

2. **Run the Visualization**:
   - Execute the following command in your terminal:
     ```bash
     python plot.py
     ```
   - The script processes each video, applying bounding boxes and marking the shuttle's position as per the CSV files.


## Dataset Details

| -/-       | Top-View Camera | Back-View Camera |
| --------- | -------- | -------- | 
| Device    | DJI Air 2S   | DJI Air 2S   |
| FPS       | 30           |    30        | 
| Bbox height & width     |  - |  -  |
| Shuttlecock location    |  -  |  -  |
| Poses (2 players)  |  -  |  -  |



## Usage
- The processed data can be downloaded from [here](https://drive.google.com/file/d/1DcaLrBW0IGFKLnvDKuqXlVz0PWTNU6Pz/view?usp=drive_link).
- Pretrained weights can be downloaded from [here](https://drive.google.com/file/d/1noNMyn0G_1Oqyg-na6vuW_SyabVQtF6W/view?usp=drive_link).
- For training and testing, please run `python main.py`, parameters can be modified in the configuration.py file
- For control area visualization, please run `python visualize.py --checkpoint_path`

## Contributing

We welcome contributions to this dataset and the visualization tool! If you have suggestions or encounter any issues, please feel free to open an issue or submit a pull request.





