# YOLO11 Object Detection with Custom Dataset

## Overview
This project demonstrates object detection using a custom dataset with YOLO11. The workflow includes data labeling, model training, and evaluation.

## Features
- Custom object detection using YOLO11.
- Data labeling with LabelStudio.
- Lightweight and efficient training on standard hardware.

## Requirements
- **Hardware**: Intel i5 processor, RTX 3050 GPU.
- **Software Environment**: Conda environment with Python 3.9.

## Installation
1. Create a Conda environment:
   ```bash
   conda create -n yolo11_env python=3.9 -y
   conda activate yolo11_env
   ```
2. Install dependencies:
   ```bash
   pip install ultralytics
   ```

## Data Preparation
1. **Data Labeling**:
   - Use [LabelStudio](https://labelstud.io/) for annotating the dataset.
   - Export the annotations in a format compatible with YOLO11.

2. **Dataset Structure**:
   Organize the dataset into the following structure:
   ```
   dataset/
   |-- train/
   |   |-- images/
   |   |-- labels/
   |-- val/
       |-- images/
       |-- labels/
   ```

## Training
1. Place the dataset in the appropriate directory.
2. Start training the YOLO11 model:
   ```bash
   python yolo.py
   ```

## Export
After training, we can export to coreML for iOS or tflite for android:
```bash
python export.py
```

## Additional Notes
- Ensure GPU drivers and CUDA are installed for optimal performance.
- I used the RTX 3050 GPU to speed up training and inference processes.

## References
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [LabelStudio Documentation](https://labelstud.io/)

