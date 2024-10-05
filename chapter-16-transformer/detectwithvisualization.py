#该程序基于ultralytics和OpenCV，可实现对视频文件或相机实时输入视频流的目标检测，并可视化
#同济大学，张林
import cv2
from ultralytics import RTDETR

# Load the rtdetr-l model
model = RTDETR('rtdetr-l.pt')

# Open the video file
#video_path = "test-data/test.mp4"
video_path = "http://admin:admin@100.74.37.173:8081"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("RT-DETR Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()