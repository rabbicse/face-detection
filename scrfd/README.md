# Face Detection using SCRFD

A fast and accurate face detection project using **SCRFD** (Sparse Convolutional Regression-based Face Detector) implemented with PyTorch. This repository includes examples for detecting faces in images and integrating the solution with modern frameworks.

## Features
- **High Accuracy**: Based on the powerful SCRFD model for precise face detection.
- **Lightweight and Fast**: Optimized for real-time applications on CPU and GPU.
- **Customizable**: Easily extendable for your own datasets and configurations.
- **Keypoint Detection**: Includes facial landmark (keypoint) extraction.
- **Frontend Support**: Compatible with React for rendering face detections.

## Requirements

- Python 3.10+
- PyTorch 2.5+
- OpenCV 4.10+
- CUDA (optional for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rabbicse/face-detection.git
   cd face-detection/scrfd
   ```
2. Set up the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Example: Detect Faces in an Image
1. Run the image demo:
   ```bash
   python src/examples/image_demo.py --image path_to_image.jpg --model models/SCRFD_500M_KPS.pth
   ```
2. **Output**: The detected faces and landmarks will be displayed.
 
## Acknowledgments
- [InsightFace](https://github.com/deepinsight/insightface): Inspiration and SCRFD model.
- [PyTorch](https://pytorch.org/): Framework used for development.
- [OpenCV](https://opencv.org/): Image processing and visualization.