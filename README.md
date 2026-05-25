# YasirVision 🚀

A Python library designed to simplify complex OpenCV + MediaPipe computer vision projects into easy-to-use APIs. Turn 300+ lines of repetitive code into clean, 20-line scripts!

## Installation

```bash
pip install yasirvision
```
*(Or clone this repository and run `pip install .`)*

## Features

YasirVision provides both a highly customizable `HandDetector` class and out-of-the-box demo functions.

### 1. Standalone Demo Features
Want to run a feature immediately without writing a `while True` loop? 

```python
import yasirvision as yv

yv.hand_tracking()          # Simple hand tracking
yv.virtual_drawing_canvas() # Draw in the air with your index finger!
yv.gesture_control()        # Modular gesture control (pinch, volume, cursor)
yv.finger_counter()         # Count how many fingers are held up
yv.air_write()              # Simplified air writing canvas
```

### 2. Custom Integration (Core API)
Use the `HandDetector` class to build your own computer vision logic in under 20 lines.

```python
import cv2
from yasirvision import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img)
    
    if lmList:
        fingers = detector.fingers_up()
        print(f"Fingers up: {fingers}")

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Recording a Demo
To record a showcase video for LinkedIn or GitHub, simply run the included `record_demo.py` script locally. 