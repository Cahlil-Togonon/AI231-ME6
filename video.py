import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

import mediapipe as mp
import time
import torch.cuda

import threading
import pyttsx3
import numpy as np

import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

# -------------------------------
# TTS Setup
# -------------------------------

def speak(text: str):

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)

            print("🔊 TTS speaking:", text)
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print("❌ TTS error:", e)

    # Start exactly once
    threading.Thread(target=_speak, daemon=True).start()

# -------------------------------
# Load YOLO model (TensorRT / ONNX / PT / TorchScript)
# -------------------------------
model_name = 'yolov8n'
dataset_name = 'AI231_dataset'
format = 'onnx'

format_extension = {
    'pytorch': 'pt',
    'onnx': 'onnx',
    'tensorrt': 'engine',
    'torchscript': 'torchscript'
}

device = 0 if torch.cuda.is_available() else 'cpu'

model = YOLO(f"./models/best.{format_extension[format]}")
print(f"Loaded {model_name} model in {format} format")


print("🔥 Warming up YOLO engine...")
dummy = np.zeros((640,640,3), dtype=np.uint8)
model(dummy, device=device, verbose=False)
print("✅ YOLO warm-up complete")

# -------------------------------
# POS Item List
# -------------------------------
ITEMS = {
    0:  {"name": "coffee_nescafe", "price": 5},
    1:  {"name": "coffee_kopiko", "price": 5},
    2:  {"name": "lucky-me-pancit-canton", "price": 10},
    3:  {"name": "Coke-in-can", "price": 50},
    4:  {"name": "alaska_milk", "price": 120},
    5:  {"name": "Century-Tuna", "price": 35},
    6:  {"name": "VCut-Spicy-Barbeque", "price": 30},
    7:  {"name": "Selecta-Cornetto", "price": 25},
    8:  {"name": "Nestle-Yogurt", "price": 75},
    9:  {"name": "Femme-Bathroom-Tissue", "price": 130},
    10: {"name": "maya-champorado", "price": 25},
    11: {"name": "jnj-potato-chips", "price": 30},
    12: {"name": "Nivea-Deodorant", "price": 150},
    13: {"name": "UFC-Canned-Mushroom", "price": 45},
    14: {"name": "Libbys-Vienna-Sausage-can", "price": 40},
    15: {"name": "Stik-O", "price": 60},
    16: {"name": "nissin_cup_noodles", "price": 35},
    17: {"name": "dewberry-strawberry", "price": 60},
    18: {"name": "Smart-C", "price": 40},
    19: {"name": "pineapple-juice-can", "price": 35},
    20: {"name": "nestle_chuckie", "price": 50},
    21: {"name": "Delight-Probiotic-Drink", "price": 45},
    22: {"name": "Summit-Drinking-Water", "price": 15},
    23: {"name": "almond_milk", "price": 85},
    24: {"name": "Piknik", "price": 30},
    25: {"name": "Bactidol", "price": 15},
    26: {"name": "head&shoulders_shampoo", "price": 110},
    27: {"name": "irish-spring-soap", "price": 130},
    28: {"name": "c2_na_green", "price": 35},
    29: {"name": "colgate_toothpaste", "price": 150},
    30: {"name": "555-sardines", "price": 35},
    31: {"name": "meadows_truffle_chips", "price": 40},
    32: {"name": "double-black", "price": 400},
    33: {"name": "NongshimCupNoodles", "price": 35},
}

# -------------------------------
# Global state for POS and inference status
# -------------------------------
pos_items = {}
running = False
last_count = 0

last_OD_time = 0
last_gesture_time = 0
COOLDOWN_PERIOD = 2.0  # seconds
GESTURE_RESTART = 6.0  # seconds
OD_running = False
GESTURE_running = True

last_gesture_event = True

# -------------------------------
# Mediapipe Hands for gesture recognition
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def is_open_hand(landmarks):
    # finger tips indices
    tips = [4, 8, 12, 16, 20]

    open_fingers = 0
    for tip in tips[1:]:   # ignore thumb for now
        if landmarks[tip].y < landmarks[tip - 2].y:  # fingertip above knuckle
            open_fingers += 1

    return open_fingers >= 4   # All 4 fingers extended

def is_closed_fist(landmarks):
    # finger tips indices
    tips = [4, 8, 12, 16, 20]

    closed_fingers = 0
    for tip in tips[1:]:   # ignore thumb for now
        if landmarks[tip].y > landmarks[tip - 2].y:  # fingertip below knuckle
            closed_fingers += 1

    return closed_fingers >= 4   # All 4 fingers closed

# ============================================================
#   BASIC CAMERA STREAM (NO INFERENCE)
# ============================================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
    finally:
        cap.release()


# ============================================================
#   INFERENCE STREAM (YOLO)
# ============================================================
def generate_inference_frames():
    global running, pos_items, last_OD_time, last_gesture_time, OD_running, GESTURE_running, COOLDOWN_PERIOD, GESTURE_RESTART, device

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    model_inference_time = 0
    model_fps = 0

    current_gesture = "None"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()

            if OD_running:
                results = model(frame, device=device, verbose=False)
                model_inference_time = time.time() - now
                
                annotated = results[0].plot()
                frame = annotated

                if last_OD_time and now - last_OD_time > GESTURE_RESTART:
                    GESTURE_running = True
                    OD_running = False

                elif now - last_OD_time >= COOLDOWN_PERIOD:

                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf >= 0.8:
                            item = ITEMS.get(cls)
                            if item:
                                name = item["name"]
                                price = item["price"]
                                if name in pos_items:
                                    pos_items[name]["qty"] += 1
                                else:
                                    pos_items[name] = {"qty": 1, "unit_price": price}

                                last_OD_time = now
                                print(f"✅ Detected: {name}")
                                speak(f"{name}")
            
            elif GESTURE_running:
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = results.multi_hand_landmarks[0].landmark

                    if now - last_gesture_time >= 1.0: # gesture cooldown in seconds
                        last_gesture_time = now
                        if is_open_hand(lm):
                            print("🖐️ Open hand detected → Starting inference...")
                            OD_running = True
                            GESTURE_running = False
                            last_OD_time = now
                            current_gesture = "Open Hand"

                            speak("Starting Transaction")

                        elif is_closed_fist(lm):
                            print("✊ Closed fist detected → Stopping inference...")
                            OD_running = False
                            current_gesture = "Closed Fist"

                            speak("Ending Transaction")

                        else:
                            current_gesture = "None"


            # -----------------------------
            # DRAW STATUS HEADER
            # -----------------------------
            status_text = ""
            color = (0,0,0)

            if OD_running:
                status_text = "YOLO Model: ACTIVE"
                color = (0, 255, 0)
            else:
                status_text = f"Hand Gesture: {current_gesture}"
                # orange color
                color = (0, 165, 255)

            # Draw header text
            cv2.putText(
                frame,
                status_text,
                (90, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            fps_counter += 1
            now_fps = time.time()
            elapsed = now_fps - fps_time
            update_speed = 0.5

            # Update FPS every 1 second
            if elapsed >= update_speed:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = now_fps
                model_fps = 1.0 / model_inference_time if model_inference_time > 0 else 0

            # Draw Model FPS
            cv2.putText(
                frame,
                f"Model FPS: {model_fps:.1f}",
                (frame.shape[1] - 250, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Draw Camera FPS
            cv2.putText(
                frame,
                f"Camera FPS: {current_fps:.1f}",
                (frame.shape[1] - 250, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            ok, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
    finally:
        cap.release()


# ============================================================
#   ROUTES
# ============================================================
@app.get("/")
def index():
    return FileResponse(INDEX_FILE)

@app.get("/pos_items")
def get_pos_items():
    # Calculate total per item and grand total
    global pos_items, last_count
    items_list = []
    grand_total = 0
    total_qty = 0
    for name, data in pos_items.items():
        total_price = data["qty"] * data["unit_price"]
        grand_total += total_price
        total_qty += data["qty"]
        items_list.append({
            "qty": data["qty"],
            "name": name,
            "unit_price": data["unit_price"],
            "total_price": total_price
        })

    new_item_detected = total_qty > last_count
    last_count = total_qty

    return JSONResponse({"items": items_list, "grand_total": grand_total, "new_item": new_item_detected})

# Start inference route
@app.get("/start")
def start_inference():
    global running, pos_items, last_count
    running = True
    return {"status": "inference started"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/gesture_event")
def get_gesture_event():
    global GESTURE_running, last_gesture_event
    if last_gesture_event != GESTURE_running:
        last_gesture_event = GESTURE_running
        event = "close" if GESTURE_running else "open"
    else: 
        event = None
    return JSONResponse({"event": event})

@app.get("/inference")
def inference_feed():
    return StreamingResponse(
        generate_inference_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
