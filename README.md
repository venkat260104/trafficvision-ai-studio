# 🚦 TrafficVision AI Studio

### AI-Powered Traffic Sign Recognition System

TrafficVision AI Studio is a machine learning-based web application designed to recognize and classify traffic signs from images. It uses a trained Convolutional Neural Network (CNN) model and a Flask backend to provide traffic sign recognition through a web interface.

## ✨ Features

- Traffic sign image classification.
- Deep learning-based CNN model.
- Flask web application.
- Live camera interface.
- Traffic sign prediction and classification.
- User-friendly web interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | CNN model development |
| YOLO11 | Computer vision model included in the project |
| Flask | Backend web framework |
| OpenCV | Image processing |
| HTML, CSS, JavaScript | Frontend development |
| NumPy | Numerical operations |

## 📁 Project Structure

```text
traffic_sign_project/
│
├── app.py
├── train_main.py
├── evaluate_model.py
├── inspect_dataset.py
├── test_flask.py
│
├── models/
│   └── traffic_sign_model.keras
│
├── dataset/
│   └── traffic sign recognication/
│
├── templates/
│   ├── index.html
│   └── live.html
│
├── yolo11n.pt
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/venkat260104/trafficvision-ai-studio.git
```

### 2. Open the project folder

```bash
cd trafficvision-ai-studio
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

## 🧠 Machine Learning Model

The project uses a trained CNN classification model saved in Keras format.

**Model file:** `models/traffic_sign_model.keras`

The model was evaluated with a reported test accuracy of approximately **94.44%**.

## 🎯 Project Objective

The main objective is to develop an AI-powered traffic sign recognition application that can classify traffic signs and demonstrate how computer vision and deep learning can support intelligent transportation systems.

## 👨‍💻 Developer

**Peddinti Venkata Naveen**

- GitHub: https://github.com/venkat260104

## 📌 Future Enhancements

- Improve real-time traffic sign detection.
- Integrate a traffic-sign-specific object detection model.
- Improve live camera recognition.
- Deploy the application to a cloud platform.

---

*Built with Python, Deep Learning, Computer Vision, and Flask.*