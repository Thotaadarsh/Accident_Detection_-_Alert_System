# 🚨 Accident Detection & Alert System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge\&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MobileNetV2%20%2B%20LSTM-red?style=for-the-badge)

### 🧠 AI-Powered Real-Time Accident Detection with Automated Emergency Alerts

</div>

---

## 📌 Overview

**AcciDetect AI** is an intelligent accident detection system that leverages **Deep Learning**, **Computer Vision**, and **Temporal Sequence Analysis** to detect road accidents from video streams in real time.

The system combines **MobileNetV2** for spatial feature extraction and **LSTM** networks for temporal understanding of vehicle movements, enabling accurate accident detection and immediate emergency notification.

🚑 Upon detecting an accident, the system:

✅ Identifies accident events in real time
✅ Captures accident snapshots
✅ Calculates confidence score
✅ Sends automated email alerts
✅ Provides accident location and timestamp information

---

## ✨ Key Features

🚗 Real-time accident detection from video streams

🧠 MobileNetV2 (ImageNet pretrained) + LSTM architecture

📩 Automatic email alert generation

📸 Accident snapshot capture

📍 Camera location tagging

📊 Confidence score estimation

⏱️ Timestamp generation

🚨 Emergency alert dispatch system

🔄 Cooldown mechanism to prevent duplicate alerts

---

## 🏗️ System Architecture

```text
Video Input
     │
     ▼
Frame Extraction
     │
     ▼
MobileNetV2 CNN
(Feature Extraction)
     │
     ▼
LSTM Network
(Temporal Analysis)
     │
     ▼
Accident Classification
     │
     ▼
Accident Detected?
     |
     │ Yes           
     ▼               
Capture Snapshot   
Send Email Alert   
Store Timestamp    
```

---

## 🧠 Deep Learning Architecture

### CNN Backbone

* MobileNetV2 (Pretrained on ImageNet)
* Transfer Learning Enabled
* Feature Vector Size: 1280

### Temporal Network

* LSTM Layer: 256 Units
* Dropout: 0.4
* LSTM Layer: 128 Units
* Dense Layer: 64 Units

### Output Classes

| Class | Description    |
| ----- | -------------- |
| 0     | Normal Driving |
| 1     | Near-Miss      |
| 2     | Accident       |

---

## 🛠️ Technologies Used

| Technology       | Purpose               |
| ---------------- | --------------------- |
| Python           | Programming Language  |
| TensorFlow/Keras | Deep Learning         |
| MobileNetV2      | Feature Extraction    |
| LSTM             | Temporal Analysis     |
| OpenCV           | Video Processing      |
| NumPy            | Numerical Computation |
| SMTP             | Email Notification    |

---

## 📂 Project Structure

```text
📦 AcciDetect-AI
│
├── main.py
├── test.mp4
├── accident.jpg
├── mobilenet_lstm.weights.h5
├── accident_model.weights.h5
├── README.md

```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AcciDetect-AI.git
cd AcciDetect-AI
```

### Create Virtual Environment

```bash
python -m venv accident_env
accident_env\Scripts\activate
```

### Install Dependencies

```bash
pip install tensorflow opencv-python numpy
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 📧 Automated Alert System

When an accident is detected, the system automatically sends an email containing:

📍 Accident Location

🕒 Date & Time

🎥 Camera ID

📊 CNN + LSTM Confidence Score

📸 Accident Snapshot

🚑 Emergency Alert Message

---

## 📸 Sample Outputs

### 🚨 Accident Detection

* Accident confidence estimation
* Real-time event detection
* Snapshot generation
* Automated emergency notification

---

## 🎯 Future Enhancements

* 🌐 Live CCTV Stream Integration
* 📱 SMS Notification Support
* 🚓 Police & Ambulance API Integration
* 🛰️ GPS-Based Location Tracking
* ☁️ Cloud Deployment
* 📊 Web Dashboard Analytics

---

## 👨‍💻 Author

### Adarsh Thota

📧 Email: [adarshthota07@gmail.com](mailto:adarshthota07@gmail.com)

🔗 GitHub: https://github.com/Thotaadarsh

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star ⭐

**Made with ❤️ using Deep Learning and Computer Vision**

</div>
