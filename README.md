### README.md 원본 (ONNXRuntime 기반 코드)

```markdown
# ONNX YOLO + HRNet Pose Estimation

This repository demonstrates how to perform object detection using YOLO and multi-person pose estimation using HRNet with ONNXRuntime. The project is designed to process video input and visualize both object detection results and human pose estimations.

## Features

- **YOLO Object Detection**: Detect objects in video frames using a YOLO ONNX model.
- **HRNet Pose Estimation**: Perform multi-person pose estimation using an HRNet ONNX model.
- **Visualization**: Draw bounding boxes, skeletons, and keypoints on video frames.
- **Real-Time Processing**: Efficient and real-time video processing.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.8+
- OpenCV
- NumPy
- ONNXRuntime
- PyTorch
- Matplotlib

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Directory Structure

```plaintext
.
├── models/
│   ├── yolov7-tiny.onnx         # Pre-trained YOLO ONNX model
│   ├── hrnet.onnx               # Pre-trained HRNet ONNX model
├── source/
│   └── le_sserafim.mp4          # Example video for processing
├── coco2017.txt                 # COCO class labels
├── main.py                      # Main script to run the video processing
├── requirements.txt             # Required Python libraries
└── README.md                    # Project documentation
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ryangsookim/onnx-yolo-hrnet-pose-estimation.git
   cd onnx-yolo-hrnet-pose-estimation
   ```

2. **Prepare the Environment**:
   Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place Models and Video**:
   - Download or place the ONNX models (`yolov7-tiny.onnx` and `hrnet.onnx`) into the `models/` directory.
   - Place your video file in the `source/` directory or update the `video_path` in the script.

4. **Run the Script**:
   Execute the script to process a video:
   ```bash
   python main.py
   ```

5. **Expected Output**:
   - The video frames will be displayed in a window with detected objects, bounding boxes, and skeletons drawn.
   - Press `q` to quit the visualization.

## Models

- **YOLOv7-Tiny**: Optimized for real-time object detection.
- **HRNet**: High-resolution network for accurate pose estimation.

## Example

The following example uses the provided `le_sserafim.mp4` video and the models in the `models/` directory.

```python
# Modify these paths in main.py
video_path = "source/le_sserafim.mp4"
YOLO_model_path = "models/yolov7-tiny.onnx"
HRNet_model_path = "models/hrnet.onnx"
```

## Results

- Detected objects are highlighted with bounding boxes.
- Human poses are visualized with keypoints and skeletons.

## Notes

- Ensure the ONNX models are downloaded and placed correctly in the `models/` folder.
- Use high-quality input videos for better pose estimation results.
- This project is for educational purposes and demonstrates the integration of ONNXRuntime for inference.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

This `README.md` provides a clear explanation of the repository's purpose, setup instructions, and usage, ensuring easy understanding for any user.
