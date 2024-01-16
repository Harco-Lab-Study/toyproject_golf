import cv2
cv2.namedWindow('Webcam Stream', cv2.WINDOW_NORMAL)
import numpy as np
from ultralytics import YOLO

from util import get_most_min_distance, draw_skeleton
import json
import os

PAIRS = [
    (5, 6),  # shoulder
    (5, 11), (11, 12), (12, 6),  # body
    (11, 15), (11, 13), (13, 15),  # left leg
    (12, 16), (12, 14), (14, 16),  # right leg
    (5, 9), (9, 7), (7, 5),  # left arm
    (6, 10), (10, 8), (8, 6)  # right arm
]

save_path = "/home/harcolab/git/toyproject_golf/dataset/"

LABEL_IDX = 0 # 0: not address
json_path = "/home/harcolab/git/toyproject_golf/dataset/not_address.json" ## json 파일 경로 : 필요에 따라 수정
# LABEL_IDX = 1 # 1: address
# json_path = "/home/harcolab/git/toyproject_golf/dataset/address.json" ## json 파일 경로 : 필요에 따라 수정




index = 0
data = []

if not os.path.exists(save_path):
    os.makedirs(save_path)
Target = [ 
            {'key' : 'Not Address', 'label' : 0}, 
            {'key' : 'Address', 'label' : 1}, 
        ]
if os.path.exists(json_path):
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
            if data: 
                index = max(entry["index"] for entry in data) + 1
    except json.JSONDecodeError:
        print(f"Error reading {json_path}. File might be empty or corrupt. Initializing a new dataset.")
else:
    print(f"{json_path} does not exist. Initializing a new dataset.")





# webcam
cap = cv2.VideoCapture(0)


# https://github.com/ultralytics/ultralytics 참고
model = YOLO('./참조/yolov8n-pose.pt')


count = 0
while True:
    # Read the frame
    ret, frame = cap.read()
    if count==0:
        prev_frame_center = (frame.shape[1]//2, frame.shape[0]//2)

    results = model(frame,verbose=False)[0]
    boxes_centers = results.boxes.xywh.cpu().numpy()
    boxes_centers = boxes_centers[:, :2]
    keypoints_xy_list = results.keypoints.cpu().xy.squeeze().numpy().tolist()


    # one person인경우
    if np.array(keypoints_xy_list).shape == (17,2):
        min_idx = get_most_min_distance(boxes_centers, prev_frame_center)
        keypoints_xy = np.array(keypoints_xy_list)
    
    # multiple person인 경우
    else:
        min_idx = get_most_min_distance(boxes_centers, prev_frame_center)
        

    # min_idx -1은 사람이 없는 경우, 0은 사람이 1명인 경우
    if min_idx == 0:
        draw_skeleton(frame, keypoints_xy, PAIRS)
        
    elif min_idx > 0:
        keypoints_xy = np.array(keypoints_xy_list)[min_idx]
        draw_skeleton(frame, keypoints_xy, PAIRS)
        

    # Display the frame
    cv2.imshow('Webcam Stream', frame)


    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    elif key==13: # enter
        filename = os.path.join(save_path, f"{index}.jpg")
        cv2.imwrite(filename, frame)
        # save
        if len(keypoints_xy) == 17:
            data.append({
                "index": index,
                "label": Target[LABEL_IDX]['label'],
                "keypoints": keypoints_xy.tolist()
            })
            index += 1
            with open(json_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Saved {json_path}.")


    count+=1
    if min_idx != -1:
        prev_frame_center = boxes_centers[min_idx]

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
