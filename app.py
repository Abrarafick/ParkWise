from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import json
import random
from datetime import datetime, timedelta
from ml_predictor import ParkingPredictor
from cv_detector import ParkingDetector

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# ── Global State ──────────────────────────────────────────────
detector = ParkingDetector()
predictor = ParkingPredictor()

parking_state = {
    "total_spots": 20,
    "spots": {},
    "last_updated": None,
    "camera_active": False,
}

reservations = {}  # id -> reservation dict
users = {
    "demo@parkwise.com": {"name": "Demo User", "password": "demo123", "bookings": []}
}

# Init spots
for i in range(1, 21):
    row = "A" if i <= 10 else "B"
    num = i if i <= 10 else i - 10
    parking_state["spots"][f"{row}{num}"] = {
        "id": f"{row}{num}",
        "row": row,
        "number": num,
        "status": "available",  # available / occupied / reserved
        "reserved_by": None,
        "confidence": 0.99,
    }

# Simulate some initial occupancy
for spot_id in ["A2", "A5", "A7", "B1", "B3", "B6", "B8"]:
    parking_state["spots"][spot_id]["status"] = "occupied"

parking_state["last_updated"] = datetime.now().isoformat()

# ── Background simulation thread ──────────────────────────────
def simulate_parking_changes():
    """Simulate real-time changes when no camera is connected."""
    while True:
        time.sleep(8)
        available = [s for s, d in parking_state["spots"].items() if d["status"] == "available"]
        occupied  = [s for s, d in parking_state["spots"].items() if d["status"] == "occupied"]
        # Random car arrives
        if available and random.random() < 0.4:
            spot = random.choice(available)
            parking_state["spots"][spot]["status"] = "occupied"
        # Random car leaves
        if occupied and random.random() < 0.35:
            spot = random.choice(occupied)
            parking_state["spots"][spot]["status"] = "available"
        parking_state["last_updated"] = datetime.now().isoformat()

sim_thread = threading.Thread(target=simulate_parking_changes, daemon=True)
sim_thread.start()

# ── Helper ────────────────────────────────────────────────────
def get_stats():
    spots = parking_state["spots"]
    total     = len(spots)
    occupied  = sum(1 for s in spots.values() if s["status"] == "occupied")
    reserved  = sum(1 for s in spots.values() if s["status"] == "reserved")
    available = total - occupied - reserved
    return {"total": total, "occupied": occupied, "reserved": reserved, "available": available,
            "occupancy_pct": round((occupied + reserved) / total * 100, 1)}

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# ── API: Spots ────────────────────────────────────────────────
@app.route("/api/spots", methods=["GET"])
def get_spots():
    return jsonify({
        "spots": list(parking_state["spots"].values()),
        "stats": get_stats(),
        "last_updated": parking_state["last_updated"],
        "camera_active": parking_state["camera_active"],
    })

@app.route("/api/stats", methods=["GET"])
def get_stats_api():
    return jsonify(get_stats())

# ── API: Reserve ──────────────────────────────────────────────
@app.route("/api/reserve", methods=["POST"])
def reserve_spot():
    data = request.json
    spot_id  = data.get("spot_id")
    user_name = data.get("user_name", "Guest")
    vehicle   = data.get("vehicle", "Unknown")
    duration  = int(data.get("duration", 60))  # minutes

    if spot_id not in parking_state["spots"]:
        return jsonify({"success": False, "error": "Spot not found"}), 404

    spot = parking_state["spots"][spot_id]
    if spot["status"] != "available":
        return jsonify({"success": False, "error": f"Spot {spot_id} is not available"}), 400

    res_id = f"RES{int(time.time())}"
    expires = (datetime.now() + timedelta(minutes=duration)).isoformat()

    reservations[res_id] = {
        "id": res_id,
        "spot_id": spot_id,
        "user_name": user_name,
        "vehicle": vehicle,
        "created_at": datetime.now().isoformat(),
        "expires_at": expires,
        "duration": duration,
        "status": "active",
        "amount": round(duration / 60 * 50, 2),  # BDT 50/hr
    }

    parking_state["spots"][spot_id]["status"] = "reserved"
    parking_state["spots"][spot_id]["reserved_by"] = res_id

    return jsonify({"success": True, "reservation": reservations[res_id]})

@app.route("/api/cancel/<res_id>", methods=["POST"])
def cancel_reservation(res_id):
    if res_id not in reservations:
        return jsonify({"success": False, "error": "Reservation not found"}), 404
    res = reservations[res_id]
    spot_id = res["spot_id"]
    parking_state["spots"][spot_id]["status"] = "available"
    parking_state["spots"][spot_id]["reserved_by"] = None
    reservations[res_id]["status"] = "cancelled"
    return jsonify({"success": True})

@app.route("/api/reservations", methods=["GET"])
def get_reservations():
    return jsonify({"reservations": list(reservations.values())})

# ── API: ML Prediction ────────────────────────────────────────
@app.route("/api/predict", methods=["GET"])
def predict():
    hour = int(request.args.get("hour", datetime.now().hour))
    day  = int(request.args.get("day",  datetime.now().weekday()))
    predictions = predictor.predict_availability(hour, day)
    return jsonify({"predictions": predictions, "hour": hour, "day": day})

@app.route("/api/predict/chart", methods=["GET"])
def predict_chart():
    day = int(request.args.get("day", datetime.now().weekday()))
    data = []
    for h in range(7, 23):
        pred = predictor.predict_availability(h, day)
        data.append({"hour": h, "label": f"{h:02d}:00",
                      "available": pred["available_pct"],
                      "occupied": pred["occupied_pct"]})
    return jsonify({"chart_data": data})

# ── API: CV Detection (simulated) ────────────────────────────
@app.route("/api/cv/detect", methods=["POST"])
def cv_detect():
    """Accept a base64 image or use simulated detection."""
    result = detector.detect_simulated()
    # Update spots based on detection
    for i, (spot_id, is_occupied) in enumerate(result["detections"].items()):
        if spot_id in parking_state["spots"]:
            if parking_state["spots"][spot_id]["status"] != "reserved":
                parking_state["spots"][spot_id]["status"] = "occupied" if is_occupied else "available"
    parking_state["last_updated"] = datetime.now().isoformat()
    return jsonify(result)

# ── API: History (for charts) ─────────────────────────────────
@app.route("/api/history", methods=["GET"])
def history():
    now = datetime.now()
    hist = []
    for i in range(24, 0, -1):
        t = now - timedelta(hours=i)
        base = 12 + 6 * np.sin((t.hour - 8) * np.pi / 12)
        base = max(2, min(18, base))
        occ = int(base + random.gauss(0, 1.5))
        occ = max(0, min(20, occ))
        hist.append({
            "time": t.strftime("%H:%M"),
            "occupied": occ,
            "available": 20 - occ,
            "timestamp": t.isoformat()
        })
    return jsonify({"history": hist})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
