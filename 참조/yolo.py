import cv2
cv2.namedWindow('Webcam Stream', cv2.WINDOW_NORMAL)
from ultralytics import YOLO



# Open the webcam
cap = cv2.VideoCapture(1)


# https://github.com/ultralytics/ultralytics
model = YOLO('./yolov8n-pose.pt')


while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    results = model(frame)
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Webcam Stream', annotated_frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
