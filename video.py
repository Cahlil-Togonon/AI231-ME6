import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import time
import threading
import os
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

with open('./classes.json', 'r') as f:
    ITEMS = json.load(f)

### global variables
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

class AudioManager:
    def __init__(self):
        try:
            from queue import Queue
        except ImportError:
            raise ImportError("pyttsx3 package is not installed. Please install it to use TTS features.")
        
        self.speech_queue = Queue()

        self.worker_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.worker_thread.start()
        return

    def tts_worker(self):
        try:
            import pyttsx3
        except ImportError:
            raise ImportError("pyttsx3 package is not installed. Please install it to use TTS features.")

        while True:
            text = self.speech_queue.get()
            try:
                print("🗣 SPEAK:", text)
                engine = pyttsx3.init()
                engine.setProperty('rate', 175)
                engine.setProperty('volume', 1.0)
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.say(text)
                engine.runAndWait()
                del engine

            except Exception as e:
                print("❌ TTS error:", e)
            self.speech_queue.task_done()

    def speak(self, text):
        if not text:
            return
        self.speech_queue.put(str(text))

class YOLO_Model:
    def __init__(self, model_name: str = 'yolov8n', format: str = 'onnx'):
        try:
            from ultralytics import YOLO
            import torch.cuda
        except ImportError:
            raise ImportError("ultralytics/torch package is not installed. Please install it to use YOLO features.")

        format_extension = {
            'pytorch': 'pt',
            'onnx': 'onnx',
            'tensorrt': 'engine',
            'torchscript': 'torchscript'
        }

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(f"./models/best.{format_extension[format]}")
        print(f"Loaded {model_name} model in {format} format")

        self.warmup()
        return

    def warmup(self):
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy package is not installed. Please install it to use YOLO features.")

        print("🔥 Warming up YOLO engine...")
        dummy = np.zeros((640,640,3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
        print("✅ YOLO warm-up complete")

class Gesture_Model:
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("mediapipe package is not installed. Please install it to use Gesture features.") 

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        return

    def is_open_hand(self, landmarks):
        # finger tips indices
        tips = [4, 8, 12, 16, 20]

        open_fingers = 0
        for tip in tips[1:]:   # ignore thumb
            if landmarks[tip].y < landmarks[tip - 2].y:  # fingertip above knuckle
                open_fingers += 1

        return open_fingers >= 4   # Open hand (all 4 fingers extended)

    def is_closed_fist(self, landmarks):
        # finger tips indices
        tips = [4, 8, 12, 16, 20]

        closed_fingers = 0
        for tip in tips[1:]:    # ignore thumb
            if landmarks[tip].y > landmarks[tip - 2].y:  # fingertip below knuckle
                closed_fingers += 1

        return closed_fingers >= 4   # Closed fist (all 4 fingers curled)

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        return None

    def __del__(self):
        self.cap.release()

### Basic Camera Stream
def generate_frames():
    
    camera = Camera()
    cap = camera.cap

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

### Camera Stream with Inference
def generate_inference_frames():
    global running, pos_items, last_OD_time, last_gesture_time, OD_running, GESTURE_running, COOLDOWN_PERIOD, GESTURE_RESTART, device

    CAMERA = Camera()
    cap = CAMERA.cap

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    YOLO_MODEL = YOLO_Model(model_name='yolov8n', format='onnx')
    GESTURE_MODEL = Gesture_Model()
    AUDIO_HANDLER = AudioManager()

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
                results = YOLO_MODEL.model(frame, device=YOLO_MODEL.device, verbose=False)
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
                            item = ITEMS.get(str(cls), None)
                            if item:
                                name = item["name"]
                                price = item["price"]
                                if name in pos_items:
                                    pos_items[name]["qty"] += 1
                                else:
                                    pos_items[name] = {"qty": 1, "unit_price": price}

                                last_OD_time = now
                                print(f"✅ Detected: {name}")
                                AUDIO_HANDLER.speak(f"{name}")
            
            elif GESTURE_running:
                results = GESTURE_MODEL.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        GESTURE_MODEL.mp_draw.draw_landmarks(frame, hand_landmarks, GESTURE_MODEL.mp_hands.HAND_CONNECTIONS)
                    lm = results.multi_hand_landmarks[0].landmark

                    if now - last_gesture_time >= 2.0: # gesture cooldown in seconds
                        last_gesture_time = now
                        if GESTURE_MODEL.is_open_hand(lm):
                            print("🖐️ Open hand detected → Starting inference...")
                            OD_running = True
                            GESTURE_running = False
                            last_OD_time = now
                            current_gesture = "Open Hand"

                            AUDIO_HANDLER.speak("Starting Transaction")

                        elif GESTURE_MODEL.is_closed_fist(lm):
                            print("✊ Closed fist detected → Stopping inference...")
                            OD_running = False
                            current_gesture = "Closed Fist"

                            AUDIO_HANDLER.speak("Ending Transaction")

                else:
                    current_gesture = "None"


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
            cv2.putText(frame,
                status_text,
                (90, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            fps_counter += 1
            elapsed = now - fps_time

            # Update FPS every 0.5 second
            if elapsed >= 0.5:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = now
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
