import numpy as np
import cv2

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