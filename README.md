# Drone_BD_ControlArea

This repository presents a novel method for estimating control area probability maps in doubles badminton, utilizing a unique dataset of top-view and back-view drone videos from badminton matches. This dataset is ideally suited for projects in sports analytics, motion tracking, and computer vision applications.

Ding, Ning, et al. ["Estimation of control area in badminton doubles with pose information from top and back view drone videos."](https://doi.org/10.1007/s11042-023-16362-1) *Multimedia Tools and Applications* (2023). 

<div align="center">
  <img src="https://github.com/Ning-D/Drone_BD_ControlArea/blob/main/visual/Rally.gif" alt="Description of the second GIF" width="300">
  <img src="https://github.com/Ning-D/Drone_BD_ControlArea/blob/main/visual/Estimation.gif" alt="Estimation of control area in a rally" width="400">
  
</div>






## Drone Video Dataset Access

### Study Videos and Corresponding CSV Files
The drone videos utilized in this study, along with their corresponding CSV files, can be accessed via the following Google Drive link:
- [Download Study Drone Video Dataset](https://drive.google.com/drive/folders/1sIKIDLjyhccO_y6gXeaIkr_1gu1o0vYw?usp=drive_link) (`Study_Videos` and `datacsv`)

### Complete Drone Video Collection
Additionally, the complete set of drone videos, encompassing all rallies, is available in the `All_videos` folder:
- [Download Complete Drone Video Collection](https://drive.google.com/drive/folders/1TgFaTvqqi4GzoKerhBaRKyUMkDlRJ-HZ?usp=sharing) (`All_videos`)


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

Notice: We have only labeled the shuttle position for hit/drop samples that are used in this study. Additionally, for drop cases, the shuttle position has been labeled only when it drops in the opponent's court. To obtain the shuttle position at each frame, please use the tracking algorithm or labeling tool available in [TrackNet](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2/Track) or [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2).

## Dataset Details

| -/-       | Top-View Camera | Back-View Camera |
| --------- | -------- | -------- | 
| Device    | DJI Air 2S   | DJI Air 2S   |
| FPS       | 30           |    30        | 
| Bounding Box     |  :heavy_check_mark:  |  :heavy_check_mark:   |
| Shuttlecock location    |  :heavy_check_mark:   |  :heavy_check_mark:   |
| Poses (2 players)  |  -  |  -  |



## Usage
- The processed data can be downloaded from [here](https://drive.google.com/file/d/1DcaLrBW0IGFKLnvDKuqXlVz0PWTNU6Pz/view?usp=drive_link).
- Pretrained weights can be downloaded from [here](https://drive.google.com/file/d/1noNMyn0G_1Oqyg-na6vuW_SyabVQtF6W/view?usp=drive_link).
- For training and testing, parameters can be modified in the configuration.py
  ```bash
  python main.py
  ```
- For control area visualization, please run
  ```bash
  python visualize.py --checkpoint_path
  ```
## Contributing

We welcome contributions to this dataset and the visualization tool! If you have suggestions or encounter any issues, please feel free to open an issue or submit a pull request.





