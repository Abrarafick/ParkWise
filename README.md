# 🅿️ ParkWise — Smart Parking System

> Real-Time Parking Detection & Reservation Using Computer Vision and Deep Learning

**Daffodil International University | Software Engineering Department**

---

## 📸 Demo

Open `templates/index.html` in a browser **or** run the full backend:

```bash
cd backend
python app.py
# Visit: http://localhost:5000
```

---

## 🏗️ Project Structure

```
parkwise/
├── backend/
│   ├── app.py            ← Flask REST API (main server)
│   ├── cv_detector.py    ← OpenCV parking spot detection
│   └── ml_predictor.py   ← Random Forest availability predictor
├── templates/
│   └── index.html        ← Full web dashboard (user + admin)
├── webcam_detect.py      ← Live webcam CV script (run on laptop)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Laptop Demo)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the backend
```bash
cd backend
python app.py
```

### 3. Open the dashboard
```
http://localhost:5000
```

### 4. (Optional) Run live webcam detection
```bash
python webcam_detect.py
```

---

## ✨ Features

| Feature | Tech Used |
|---------|-----------|
| Real-time parking map | OpenCV + Flask API |
| Spot availability prediction | Random Forest (Scikit-Learn) |
| Spot reservation + receipt | Flask REST API |
| Admin booking management | JavaScript + Chart.js |
| Live webcam detection | OpenCV background subtraction |
| 24-hour forecast chart | ML model + Chart.js |
| Simulated mode (no webcam) | Built-in simulation |

---

## 🧠 ML Model

- **Algorithm**: Random Forest Regressor (100 trees)
- **Features**: Hour of day, Day of week, Month, Weekend flag
- **Target**: Occupancy percentage
- **Training data**: 2000 synthetic samples with realistic patterns
- **Accuracy**: ~91% confidence on peak hours

---

## 📷 CV Detection

- **Method**: OpenCV Background Subtraction (MOG2)
- **Input**: Webcam / CCTV / IP camera
- **Output**: Per-spot occupied/free classification
- **Grid**: 20 spots (2 rows × 10 columns)
- **Upgrade path**: Replace with YOLOv8 for production

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/spots` | All spots + live stats |
| POST | `/api/reserve` | Reserve a spot |
| POST | `/api/cancel/<id>` | Cancel reservation |
| GET | `/api/predict` | ML availability prediction |
| GET | `/api/predict/chart` | 24-hr forecast data |
| POST | `/api/cv/detect` | Run CV detection |
| GET | `/api/history` | 24-hour occupancy history |
| GET | `/api/reservations` | All bookings |

---

## 🔧 Hardware Setup (Laptop Demo)

```
Laptop Webcam
     │
     ▼
webcam_detect.py (OpenCV)
     │  HTTP POST every 5s
     ▼
Flask Backend (localhost:5000)
     │
     ▼
Web Dashboard (Browser)
```

**For judges:** Run both `backend/app.py` and `webcam_detect.py` simultaneously, then open the browser dashboard.

---

## 📚 References

1. Amato et al. (2016) — Car Parking Occupancy Detection, IEEE ISCC
2. Redmon & Farhadi (2018) — YOLOv3, arXiv:1804.02767
3. de Almeida et al. (2015) — PKLot Dataset, Expert Systems with Applications
4. Ultralytics YOLOv8 (2023) — github.com/ultralytics/ultralytics
5. Scikit-Learn — scikit-learn.org

---

**Built with ❤️ for DIU SE Project Showcase**
