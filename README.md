# Road Damage Detector

A Flutter mobile application that detects and classifies road damage in real time using a **MobileNetV2** deep learning model converted to **TensorFlow Lite**. The app runs entirely on-device — no internet connection required for detection.

---

## Screenshots

<img width="300" alt="image" src="https://github.com/user-attachments/assets/83dd0a66-0f10-436e-bc41-448c81a30d63" />


---

## Features

| Feature          | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| Camera Detection | Capture road images directly from the phone camera          |
| Image Upload     | Pick any image from the gallery for analysis                |
| On-device AI     | Runs TFLite inference locally — no internet needed          |
| Bounding Boxes   | Draws labeled boxes around each detected damage area        |
| Summary Panel    | Shows total detections, count per class, and top confidence |
| Model Status     | Live indicator showing when the model is ready              |

---

## 🔍 Damage Classes

| Class            | Color     | Description                                       |
| ---------------- | --------- | ------------------------------------------------- |
| Pothole          | 🔴 Red    | Bowl-shaped holes caused by wear and water damage |
| Crack            | 🟠 Orange | Linear fractures on the road surface              |
| Surface Distress | 🟡 Yellow | General deterioration such as rutting or raveling |

---

## System Design

### DFD Level 0 — Context Diagram

```
                    ┌─────────────────────────┐
                    │                         │
Camera/Sensor ─────▶│   Road Damage Detection │─────▶ Damage Records DB
                    │         System          │
     User ─────────▶│                         │─────▶ User (Result)
                    └─────────────────────────┘
```

### DFD Level 1 — Internal Processes

```
P1 Image Capture ──▶ P2 Image Preprocessing ──▶ P3 Damage Classification ──▶ P4 Result Display
                              │                            │
                              ▼                            ▼
                      D1 Image Storage            D2 Damage Records
```

**Processes:**

- **P1 – Image Capture** — Triggered by the user via camera or gallery
- **P2 – Image Preprocessing** — Resizes and normalizes image to 640×640 for the model
- **P3 – Damage Classification** — MobileNetV2/TFLite model outputs class and confidence
- **P4 – Result Display** — Shows bounding boxes and summary to the user

**Data Stores:**

- **D1 – Image Storage** — Temporary local storage for captured images
- **D2 – Damage Records** — Stores classification results and history

---

## 🛠️ Tech Stack

| Technology      | Purpose                                  |
| --------------- | ---------------------------------------- |
| Flutter         | Cross-platform mobile UI framework       |
| Dart            | Programming language                     |
| TensorFlow Lite | On-device ML inference                   |
| MobileNetV2     | Lightweight CNN for image classification |
| YOLOv8          | Object detection architecture            |
| camera          | Flutter camera plugin                    |
| image_picker    | Gallery image selection                  |
| tflite_flutter  | TFLite Flutter integration               |
| image           | Image preprocessing                      |
| path_provider   | File system access                       |

---

## Getting Started

### Prerequisites

- Flutter SDK (3.x or higher)
- Android Studio or VS Code
- Android device or emulator (API 21+)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ahiaaa/road-damage-detector.git
   cd road-damage-detector
   ```

2. **Install dependencies**

   ```bash
   flutter pub get
   ```

3. **Add the TFLite model**

   Place your `road_damage_50epochs.tflite` file inside the `assets/` folder.
   Make sure `pubspec.yaml` includes:

   ```yaml
   flutter:
     assets:
       - assets/road_damage_50epochs.tflite
   ```

4. **Run the app**
   ```bash
   flutter run
   ```

---

## Project Structure

```
road_damage_app/
├── assets/
│   └── road_damage_50epochs.tflite   # TFLite model
├── lib/
│   └── main.dart                     # Main app code
├── android/                          # Android configuration
├── ios/                              # iOS configuration
├── pubspec.yaml                      # Dependencies
└── README.md
```

---

## Model Details

| Property             | Value                                           |
| -------------------- | ----------------------------------------------- |
| Base Architecture    | MobileNetV2                                     |
| Format               | TensorFlow Lite (.tflite)                       |
| Input Size           | 640 × 640 px                                    |
| Output Shape         | [1, 7, 8400]                                    |
| Training Epochs      | 50                                              |
| Confidence Threshold | 0.35                                            |
| IOU Threshold        | 0.50                                            |
| Classes              | 4 (Pothole, Crack, Surface Distress, No Damage) |

---

## Group Members

| Name     | Role                             |
| -------- | -------------------------------- |
| Member 1 | Introduction & Problem Statement |
| Member 2 | Features & App Walkthrough       |
| Member 3 | Technical Implementation         |
| Member 4 | Challenges & Conclusion          |

---

## License

This project is developed for academic purposes only.
