# FusionFrame

FusionFrame is basic image and video processing application built using `python`, `opencv` and the `customtkinter` library. It allows users to apply various filters, detect objects using pre-trained YOLOv3, process prictures, stored videos, and live video feeds from a webcam.

This was a weekend project as part of an extensive two months "Deep Neural Network Bootcamp" at GIKI Sawabi. 

## Table of Contents

- [FusionFrame](#fusionframe)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)

## Features

- Load and process images and videos, and live webcam.
- Apply various filters such as Gaussian Blur, Median Blur, Bilateral Filter, and more.
- Detect objects in images and videos using YOLOv3.
- View live video feed from the webcam and apply filters or object detection in real-time.
- Switch between Dark, Light, and System themes.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fusionframe.git
   cd fusionframe
- Make sure the `yolo` sub-directory contains `coco.names`, `yolov3.cfg`, and `yolov3.weights`
- If you intend to change the names, do change the paths in the YOLODetector class constructor for all three

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the application:**
   ```bash
   python fusionframe.py
   ```

2. **Features Overview:**
- Source Selection: Load images, videos, or start the live camera feed.
- Object Detection: Use YOLOv3 to detect objects in the selected source.
- Filters List: Apply various filters to the selected source.

## Contributing
Feel free to add your creativity to FusionFrame! To contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'commit message').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

