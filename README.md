# Pothole_Counter
# Pothole Detection Using YOLOv8

## Description

This project detects potholes in road videos using a custom-trained YOLOv8 model. The detected potholes are highlighted with bounding boxes and confidence scores, and the processed video is saved automatically.

## Features

* Detects potholes from video files
* Displays confidence scores
* Draws bounding boxes around potholes
* Saves the output video
* Real-time visualization

## Sample Detection

![Pothole Detection](POTHOLE_IMG.png)

*Example of pothole detection using the trained YOLOv8 model.*

## Requirements

* Python 3.x
* OpenCV
* Ultralytics YOLOv8
* CVZone


## Usage

1. Place your video inside the `Videos` folder.
2. Place your trained model inside the `yolo-weights` folder.
3. Update the video and model paths if required.
4. Run:

```bash
python pothole_detection.py
```

Press **Q** to exit.

## Output

The processed video with detected potholes will be saved as:

```text
pothole_output.mp4
```

## Technologies Used

* Python
* OpenCV
* YOLOv8
* CVZone

## Author

Md. Ayan Akhtar Khan
