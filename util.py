import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


## For Webcam.py
def get_most_min_distance(box_center, frame_center):
    if len(box_center) == 0:  # 객체가 감지되지 않으면 -1 반환
        return -1
    distance = np.linalg.norm(box_center - frame_center, axis=1)
    min_idx = np.argmin(distance)
    return min_idx


def draw_skeleton(cv2_frame, keypoints, pairs):
    print(keypoints)
    for pair in pairs:
        start_point = tuple(keypoints[pair[0]].astype(int))
        end_point = tuple(keypoints[pair[1]].astype(int))
        

        if pair in [(5, 6)]:  # shoulder
            color = (255, 0, 0)  # blue
        elif pair in [(5, 11), (11, 12), (12, 6)]:  # body
            color = (0, 255, 255)  # yellow
        elif pair in [(11, 15), (11, 13), (13, 15), (12, 16), (12, 14), (14, 16)]:  # legs
            color = (0, 0, 255)  # red
        else:  # arms
            color = (255, 255, 0)  # skyblue

        # Draw lines and circles
        cv2.line(cv2_frame, start_point, end_point, color, 2)
        cv2.circle(cv2_frame, start_point, 3, (0, 255, 0), -1)  # green circle
        cv2.circle(cv2_frame, end_point, 3, (0, 255, 0), -1)  # green circle

    # Draw circles for the face keypoints
    for i in range(5):  # Assuming 0 to 4 are face keypoints
        point = tuple(keypoints[i].astype(int))
        cv2.circle(cv2_frame, point, 3, (0, 255, 0), -1)  # green circle

    cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    return cv2_frame






## For preprocess.ipynb
def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def normalization(keypoints):
    # 만약 2차원이면 3차원으로 만들어서 처리한후, 다시 2차원으로 만들어준다.
    input_shape = keypoints.shape
    if len(input_shape) == 2:
        keypoints = keypoints.reshape(-1,17,2)
    center_head_y = keypoints[:,0,1]
    center_foot_y = (keypoints[:,15,1] + keypoints[:,16,1])/2
    center_foot_00 = (keypoints[:,15,:] + keypoints[:,16,:])/2


    # 1. 15,16번점의 중점을 (0,0)으로 만들어준다.
    keypoints = keypoints - center_foot_00.reshape(-1,1,2)

    # 2. 0번점의 Y축 값과 15,16의 Y축값의 길이를 0과 1로 만들어준다.
    height = center_head_y - center_foot_y
    keypoints= keypoints/ height.reshape(-1,1,1)


    if len(input_shape) == 2:
        keypoints = keypoints.reshape(17,2)
    return keypoints


def body_drawing(sample):
    # Head 0 ~ 4
    for i in range(5):
        plt.scatter(sample[i, 0],sample[i, 1], s=10, color='red')

    # Body
    plt.plot((sample[5, 0],sample[6, 0]),(sample[5, 1],sample[6, 1]), color='blue')
    plt.plot((sample[6, 0],sample[12, 0]),(sample[6, 1],sample[12, 1]), color='blue')
    plt.plot((sample[12, 0],sample[11, 0]),(sample[12, 1],sample[11, 1]), color='blue')
    plt.plot((sample[11, 0],sample[5, 0]),(sample[11, 1],sample[5, 1]), color='blue')

    # Left Arm
    plt.plot((sample[5, 0],sample[9, 0]),(sample[5, 1],sample[9, 1]), color='green')
    plt.plot((sample[9, 0],sample[7, 0]),(sample[9, 1],sample[7, 1]), color='green')
    plt.plot((sample[7, 0],sample[5, 0]),(sample[7, 1],sample[5, 1]), color='green')

    # Right Arm
    plt.plot((sample[6, 0],sample[10, 0]),(sample[6, 1],sample[10, 1]), color='green')
    plt.plot((sample[10, 0],sample[8, 0]),(sample[10, 1],sample[8, 1]), color='green')
    plt.plot((sample[8, 0],sample[6, 0]),(sample[8, 1],sample[6, 1]), color='green')

    # Left Leg
    plt.plot((sample[11, 0],sample[13, 0]),(sample[11, 1],sample[13, 1]), color='purple')
    plt.plot((sample[13, 0],sample[15, 0]),(sample[13, 1],sample[15, 1]), color='purple')
    plt.plot((sample[15, 0],sample[11, 0]),(sample[15, 1],sample[11, 1]), color='purple')

    # Right Leg
    plt.plot((sample[12, 0],sample[14, 0]),(sample[12, 1],sample[14, 1]), color='purple')
    plt.plot((sample[14, 0],sample[16, 0]),(sample[14, 1],sample[16, 1]), color='purple')
    plt.plot((sample[16, 0],sample[12, 0]),(sample[16, 1],sample[12, 1]), color='purple')


    plt.grid()
