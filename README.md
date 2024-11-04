# 👥 Face Mask & Social Distancing Detection System

A real-time AI-powered system that monitors face mask usage and social distancing compliance using computer vision and deep learning.

## 🎯 Project Overview

This system uses advanced computer vision techniques to:
- Detect and track faces in real-time
- Check if people are wearing masks
- Monitor social distancing between individuals
- Log compliance data for analysis

## ✨ Key Features

- **Real-time Face Detection**: Using face_recognition library
- **Mask Detection**: Deep learning model using VGG16
- **Social Distance Monitoring**: Calculates intersection areas between people
- **Person Tracking**: Assigns unique IDs to track individuals
- **Logging System**: Maintains detailed logs of violations
- **Multi-threaded Processing**: Efficient video capture and processing

## 🛠️ Technical Stack

### Core Technologies
- Python 3.9+
- OpenCV
- TensorFlow/Keras
- face_recognition
- Threading

### Key Components
- VGG16 for mask detection
- Face recognition for tracking
- Multi-threading for performance
- Logging system for monitoring

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/Amitjangir010/mask_detection.git
cd face-mask-social-distance
```

## 🚀 How to Use

1. **Setup Environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. **Prepare Dataset**:
```bash
python src/dataset_preparation.py
```

3. **Train Model**:
```bash
python src/training.py
```

4. **Run Application**:
```bash
python src/main.py
```


2. Controls:
- Press 'q' to quit
- ESC to exit fullscreen
- Space to pause/resume

## 📊 System Components

### 1. Face Detection & Tracking
```python
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
```

### 2. Mask Detection
```python
mask_prediction = detect_and_predict_mask(face, mask_model)
mask_status = 'Mask' if mask_prediction < 0.5 else 'No-Mask'
```

### 3. Social Distancing
```python
intersection_area = calculate_intersection_area(box1, box2)
if intersection_area > THRESHOLD:
    display_warning()
```

## 📁 Project Structure

```
project/
├── dataset/                     # Training data
│   ├── with_mask/              # Mask images
│   └── without_mask/           # No mask images
│
├── models/                      # Trained models
│   └── mask_detector_model.h5   # Mask detection model
│
├── src/                        # Source code
│   ├── main.py                 # Main application
│   ├── mask_detection.py       # Mask detection logic
│   ├── social_distancing.py    # Distance monitoring
│   ├── training.py            # Model training
│   └── utils.py               # Helper functions
│
├── logs/                       # Log files
│   └── logs.txt               # Application logs
│
├── requirements.txt            # Dependencies
└── README.md                  # Documentation
```

## 🔍 Features in Detail

### Mask Detection
- VGG16 based model
- Binary classification (mask/no-mask)
- Real-time processing
- High accuracy prediction

### Social Distancing
- Dynamic distance calculation
- Intersection area analysis
- Real-time warnings
- Configurable thresholds

### Person Tracking
- Unique ID assignment
- Face encoding comparison
- Persistent tracking
- Status history

## 📊 Logging System

- Detailed event logging
- Status changes tracking
- Warning records
- System diagnostics

---
Made with ❤️ by Amit Jangir
