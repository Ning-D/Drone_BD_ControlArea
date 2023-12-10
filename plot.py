import os
import pandas as pd
import cv2
import numpy as np
skipped_videos = []

# Base directories for videos and CSV files
base_video_dir = './Videos/'  # Points to the Videos folder
base_csv_dir = './datacsv/'  # Points to the datacsv folder
output_video_base_dir = './processed_videos'  # Output directory

# Ensure the base output directory exists
os.makedirs(output_video_base_dir, exist_ok=True)

# Define colors for each player's bounding box in BGR format
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Function to draw bounding boxes and shuttle position
def draw_on_frame(frame, row, skipped_videos, video_name):
    # Draw bounding boxes for all players
    for player in range(1, 5):
        top_left_x = row.get(f'p{player}x_output0')
        top_left_y = row.get(f'p{player}y_output0')
        bottom_right_x = row.get(f'p{player}x_output2')
        bottom_right_y = row.get(f'p{player}y_output2')
        # Check if any coordinate is NaN and skip drawing if so
        if any(np.isnan(coord) for coord in [top_left_x, top_left_y, bottom_right_x, bottom_right_y]):
            continue
        if top_left_x is None or top_left_y is None or bottom_right_x is None or bottom_right_y is None:
            continue

        color = colors[player - 1]
        cv2.rectangle(frame, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color=color, thickness=5)

    # Check for shuttle position
    shuttle_x = row.get('shuttle_x')
    shuttle_y = row.get('shuttle_y')
    if shuttle_x is not None and shuttle_y is not None:
        cv2.circle(frame, (int(shuttle_x), int(shuttle_y)), radius=10, color=(0, 0, 255), thickness=-1)
    else:
        # Add the video name to the skipped list if shuttle position is missing
        if video_name not in skipped_videos:
            skipped_videos.append(video_name)

# Function to process videos in a given rally folder
def process_rally_videos(rally_path, rally_csv_path, rally_output_dir, skipped_videos):
    for video_file in os.listdir(rally_path):
        if video_file == '1.mp4':
            video_path = os.path.join(rally_path, video_file)
            video_name = os.path.basename(video_path)
            csv_file = video_file.replace('.mp4', '.csv')
            csv_path = os.path.join(rally_csv_path, csv_file)

            data = pd.read_csv(csv_path)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = os.path.join(rally_output_dir, video_file)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                row = data[data['frame'] == frame_number]
                if not row.empty:
                    row = row.iloc[0]
                    draw_on_frame(frame, row, skipped_videos, video_name)  # Pass skipped_videos and video_name



                out.write(frame)
                frame_number += 1

            cap.release()
            out.release()

# Process videos for each BDxx directory
for bd_folder in os.listdir(base_video_dir):
    bd_path = os.path.join(base_video_dir, bd_folder)
    bd_csv_path = os.path.join(base_csv_dir, bd_folder)
    bd_output_dir = os.path.join(output_video_base_dir, bd_folder)
    os.makedirs(bd_output_dir, exist_ok=True)

    if os.path.isdir(bd_path):
        for rally_folder in os.listdir(bd_path):
            rally_path = os.path.join(bd_path, rally_folder)
            rally_csv_path = os.path.join(bd_csv_path, rally_folder)
            rally_output_dir = os.path.join(bd_output_dir, rally_folder)
            os.makedirs(rally_output_dir, exist_ok=True)

            if os.path.isdir(rally_path):
                process_rally_videos(rally_path, rally_csv_path, rally_output_dir,skipped_videos)

print("Processing complete.")
print("Videos with skipped shuttle positions:", skipped_videos)