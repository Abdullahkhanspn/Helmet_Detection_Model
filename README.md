# Helmet Detection Model

https://github.com/user-attachments/assets/aebd363c-d980-4f50-a49d-abced7fecd55

## Helmet detection model Colab code -
https://colab.research.google.com/drive/19J2oMdi9QGJrRsurrOzjXtwQFPDRdY0D?usp=sharing

## Overview
This project implements a machine learning model that detects whether a person riding a motorbike is wearing a helmet. The model is built using YOLO (You Only Look Once) architecture and focuses specifically on helmet detection for motorbike riders, excluding other vehicles like cars or bicycles.

## Assignment Context
This project was developed as part of a machine learning hiring assignment with the following deliverables:
- Trained model capable of detecting helmet usage
- Complete implementation with multiple inference modes
- GitHub repository with code and model weights
- README with setup and usage instructions

## Project Structure
```
Helmet Detection Model/
├── best.pt                          # Trained YOLO model weights (22.5MB)
├── Helmet_Detection_DataSet/        # Training dataset (8,027 items)
├── Model/                           # Model training artifacts
├── output/                          # Detection results output directory
├── venv/                            # Virtual environment
├── detect_single_image.py           # Single image detection script
├── detect_video.py                  # Video processing script
├── detect_webcam.py                 # Real-time webcam detection
├── frontend.py                      # GUI application
├── model_test.py                    # Model testing script
├── test_multiple.py                 # Batch testing script
├── bike_1.mp4                       # Sample video for testing
├── test_image.jpeg                  # Sample image for testing
├── logo.JPG                         # Application logo
└── requirements.txt                 # Python dependencies
```

### Model Weights Details
## Model File Information
- File Name: best.pt
- File Size: 22.5 MB (22,519,082 bytes)
- Architecture: YOLOv8s (Ultralytics)
- Input Size: 640x640 pixels
- Classes: 2 (with helmet, without helmet)
- Created: April 7, 2026
  
### Model Specifications

- Framework: Ultralytics YOLO
- Model Type: YOLOv8s (small variant for faster inference)
- Training Dataset: 6,230 annotated images
- Confidence Threshold: 0.5 (adjustable)
- Output Format: Bounding boxes with confidence scores
  
## Model Performance
Detection Classes:
Green bounding boxes: with helmet
Red bounding boxes: without helmet
Real-time Capable: ~30 FPS on GPU
Inference Speed: Optimized for real-time applications

Model Usage
The model weights (best.pt) are ready for inference and can be loaded using:

## Key Features
### **Core Detection Capabilities**
- **Helmet Detection**: Identifies riders wearing helmets vs. without helmets
- **Real-time Processing**: Supports live webcam detection
- **Batch Processing**: Handles video files with frame-by-frame analysis
- **High Accuracy**: Trained model with confidence scoring

### **Multiple Interface Options**
1. **Command Line Scripts**: For automated processing
2. **GUI Application**: User-friendly interface with modern design
3. **Webcam Integration**: Real-time detection capabilities

### **Detection Classes**
- `with_helmet`: Rider is wearing a helmet (Green bounding box)
- `without_helmet`: Rider is not wearing a helmet (Red bounding box)

## Installation & Setup
### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- 8GB+ RAM

### Environment Setup
1. **Clone/Download the Repository**
   ```bash
   git clone <repository-url>
   cd "Helmet Detection Model"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
The project uses the following key libraries:
- `ultralytics` - YOLO model implementation
- `opencv-python` - Computer vision and image processing
- `tkinter` - GUI framework (usually included with Python)

## Usage Instructions
### 1. GUI Application (Recommended)
Launch the user-friendly GUI application:

```bash
python frontend.py
```

**GUI Features:**
- **Modern Interface**: Dark theme with intuitive controls
- **Multiple Detection Modes**:
  - Detect Image: Process single images
  - Process Video: Analyze video files
  - Start Webcam: Real-time detection
  - Stop: Terminate webcam feed
- **Live Status Updates**: Real-time processing feedback
- **Visual Results**: Bounding boxes with confidence scores

### 2. Command Line Scripts
#### Single Image Detection
```bash
python detect_single_image.py
```
- Update the `image_path` variable in the script
- Results saved to `output/image_results/`

#### Video Processing
```bash
python detect_video.py
```
- Update the `video_path` variable in the script
- Outputs processed video with detection overlays
- Shows violation and safety counts

#### Webcam Detection
```bash
python detect_webcam.py
```
- Real-time helmet detection using webcam
- Press 'q' to quit

#### Model Testing
```bash
python model_test.py
```
- Basic model loading test

#### Batch Testing
```bash
python test_multiple.py
```
- Test multiple images at once

## Model Architecture & Training
### Dataset
- **Source**: Custom helmet detection dataset
- **Size**: 6,230 images
- **Link**: [https://app.roboflow.com/abdullahs-workspace-kmsho/bike-helmet-detection-6pxyk-iyibj]
- **Classes**: 2 (with_helmet, without_helmet)
- **Annotation Format**: YOLO format

### Model Specifications
- **Architecture**: YOLOv8s (Ultralytics)
- **Input Size**: 640x640 pixels
- **Model File**: `best.pt` (22.5MB)
- **Confidence Threshold**: 0.5 (adjustable)

### Training Process
1. **Data Collection**: Curated dataset focusing on motorbike riders
2. **Annotation**: Labeled bounding boxes for helmet/no-helmet
3. **Model Training**: YOLOv8 with transfer learning
4. **Evaluation**: Precision, Recall, F1-score metrics
5. **Optimization**: Model fine-tuning for accuracy

## Performance Metrics
### Detection Accuracy
- **Precision**: High accuracy in helmet detection
- **Recall**: Effective identification of violations
- **F1-Score**: Balanced performance metric
- **Inference Speed**: Real-time capable (~30 FPS on GPU)

### Output Format
- **Bounding Boxes**: Color-coded (Green for helmet, Red for no helmet)
- **Confidence Scores**: Displayed with each detection
- **Frame Statistics**: Violation counts and safety counts
- **Export Options**: Save results as images/videos

## File Formats & Outputs
### Input Formats Supported
- **Images**: JPG, JPEG, PNG, BMP
- **Videos**: MP4, AVI, MOV, MKV
- **Live Feed**: Webcam (USB/IP cameras)

### Output Formats
- **Annotated Images**: Saved with bounding boxes and labels
- **Processed Videos**: MP4 format with real-time detection overlays
- **Statistics**: Console output with detection counts and confidence scores

## Troubleshooting
### Common Issues

1. **Model Loading Error**
   - Ensure `best.pt` is in the root directory
   - Check ultralytics installation: `pip install ultralytics`

2. **GPU Not Detected**
   - Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - Model will fall back to CPU if GPU unavailable

3. **Webcam Access Issues**
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Try different camera indices if multiple cameras available

4. **Memory Issues**
   - Reduce input image size in model configuration
   - Close other memory-intensive applications
   - Use CPU inference if GPU memory is limited

### Performance Optimization
- **GPU Acceleration**: Ensure CUDA is properly installed
- **Batch Processing**: Process multiple images simultaneously
- **Model Quantization**: Consider model optimization for deployment
- **Confidence Threshold**: Adjust based on use case requirements

## API Reference
### Key Functions

#### `detect_image(image_path, confidence=0.5)`
- **Purpose**: Process single image for helmet detection
- **Parameters**: 
  - `image_path`: Path to input image
  - `confidence`: Detection confidence threshold (0-1)
- **Returns**: YOLO detection results

#### `detect_video(video_path, confidence=0.5, show_preview=False)`
- **Purpose**: Process video file with frame-by-frame detection
- **Parameters**:
  - `video_path`: Path to input video
  - `confidence`: Detection confidence threshold
  - `show_preview`: Show processing preview window
- **Output**: Processed video with detection overlays

### Code Style
- Follow PEP 8 Python style guidelines
- Add comprehensive comments for complex functions
- Include docstrings for all public functions
- Use meaningful variable names

**Note**: This project was developed as part of a machine learning hiring assignment. The model demonstrates proficiency in computer vision, deep learning, and software development best practices.
