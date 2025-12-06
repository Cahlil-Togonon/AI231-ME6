import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import time
import threading
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

class MainApplication:
    def __init__(self):
        try:
            import json
            with open('./classes.json', 'r') as f:
                self.class_items = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading class items: {e}")

        self.pos_items = {}
        self.last_count = 0
        self.total_qty = 0
        self.grand_total = 0.0
        self.state = 'pre-transaction'  # other states: 'in-transaction', 'post-transaction', 'checkout'
        self.ready = False
        self.YOLO_ready = False
        self.GESTURE_ready = False

        global AUDIO_HANDLER
        AUDIO_HANDLER = AudioManager()

        threading.Thread(target=self.initialize_YOLO_model, daemon=True).start()
        threading.Thread(target=self.initialize_gesture_model, daemon=True).start()

        while not (self.YOLO_ready and self.GESTURE_ready):
            time.sleep(0.1)

        self.ready = True

        return
    
    def initialize_YOLO_model(self):
        global YOLO_MODEL
        YOLO_MODEL = YOLO_Model(model_name='yolov8n', format='onnx')
        self.YOLO_ready = True
        return
    
    def initialize_gesture_model(self):
        global GESTURE_MODEL
        GESTURE_MODEL = Gesture_Model()
        self.GESTURE_ready = True
        return
    
    def get_app_state(self):
        return self.state
    
    def set_app_state(self, new_state: str):
        self.state = new_state
        return

    def add_item(self, cls: int):
        item = self.class_items.get(str(cls), None)
        name = None
        if item:
            name = item["name"]
            price = item["price"]
            if name in self.pos_items:
                self.pos_items[name]["qty"] += 1
                self.pos_items[name]["total_price"] += price
            else:
                self.pos_items[name] = {"qty": 1, "unit_price": price, "total_price": price}
        
        self.total_qty += 1
        self.grand_total += price

        return name

    def new_item_added(self):
        return self.total_qty > self.last_count

    def clear_items(self):
        self.pos_items.clear()
        self.last_count = 0
        self.total_qty = 0
        self.grand_total = 0.0
        self.state = 'pre-transaction'

        AUDIO_HANDLER.speak("Restarted. Welcome!")
        return

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
                engine.setProperty('pitch', 70)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                # del engine

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

        self.last_OD_time = 0
        self.GESTURE_RESTART = 5.0  # seconds
        self.COOLDOWN_PERIOD = 2.0  # seconds

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

    def inference(self, frame, time_now, MAIN_APP: MainApplication, AUDIO_HANDLER: AudioManager):
        results = self.model(frame, device=self.device, verbose=False)
        model_inference_time = time.time() - time_now

        annotated = results[0].plot()
        frame = annotated

        if MAIN_APP.pos_items and time_now - self.last_OD_time >= self.GESTURE_RESTART:
            MAIN_APP.set_app_state('post-transaction')
            AUDIO_HANDLER.speak("Show a closed fist to end the transaction")
            return frame, model_inference_time

        elif time_now - self.last_OD_time >= self.COOLDOWN_PERIOD:

            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.8:
                    name = MAIN_APP.add_item(cls)
                    if name:
                        self.last_OD_time = time_now

                        print(f"✅ Detected: {name}")
                        AUDIO_HANDLER.speak(f"{name}")
            
        return frame, model_inference_time

class Gesture_Model:
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("mediapipe package is not installed. Please install it to use Gesture features.") 

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.current_gesture = "None"
        self.closed_fist_start_time = None
        self.open_hand_start_time = None
        self.GESTURE_HOLD_DURATION = 2.0  # seconds
        return

    def check_gesture_hold(self, gesture: str, target_gesture: str, start_time_attr: str) -> bool:
        """Check if a gesture has been held for the required duration."""
        if gesture == target_gesture:
            start_time = getattr(self, start_time_attr)
            if start_time is None:
                setattr(self, start_time_attr, time.time())
            else:
                elapsed = time.time() - start_time
                if elapsed >= self.GESTURE_HOLD_DURATION:
                    setattr(self, start_time_attr, None)
                    return True
        else:
            setattr(self, start_time_attr, None)
        return False

    def inference(self, frame, MAIN_APP: MainApplication, AUDIO_HANDLER: AudioManager):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_gesture = "None"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            lm = results.multi_hand_landmarks[0].landmark

            if self.is_open_hand(lm):
                # print("🖐️ Open hand detected")
                current_gesture = "Open Hand"
            elif self.is_closed_fist(lm):
                # print("✊ Closed fist detected")
                current_gesture = "Closed Fist"
            else:
                current_gesture = "None"


            if MAIN_APP.get_app_state() == 'pre-transaction':
                if self.check_gesture_hold(current_gesture, "Open Hand", "open_hand_start_time"):
                    print("🖐️ Open hand detected")
                    MAIN_APP.set_app_state('in-transaction')
                    AUDIO_HANDLER.speak("Starting Transaction.")
                    
            elif MAIN_APP.get_app_state() == 'post-transaction':
                if self.check_gesture_hold(current_gesture, "Open Hand", "open_hand_start_time"):
                    print("✊ Closed fist detected")
                    MAIN_APP.set_app_state('in-transaction')
                    AUDIO_HANDLER.speak("Resuming Transaction.")
                    YOLO_MODEL.last_OD_time = time.time()           ## kinda bad, but works

                elif self.check_gesture_hold(current_gesture, "Closed Fist", "closed_fist_start_time"):
                    MAIN_APP.set_app_state('checkout')
                    AUDIO_HANDLER.speak("Checkout initiated.")
            
            else:
                raise ValueError("Invalid app state for gesture inference")

        return frame, current_gesture

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

MAIN_APP = MainApplication()

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
    global MAIN_APP

    CAMERA = Camera()
    cap = CAMERA.cap

    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    AUDIO_HANDLER.speak("Welcome!")

    while not MAIN_APP.ready:
        time.sleep(0.1)

    AUDIO_HANDLER.speak("System Ready.")

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

            time_now = time.time()

            app_state = MAIN_APP.get_app_state()

            status_text = ""
            color = (0,0,0)

            match app_state:
                case 'pre-transaction':
                    frame, current_gesture = GESTURE_MODEL.inference(frame, MAIN_APP, AUDIO_HANDLER)
                    status_text = f"Hand Gesture: {current_gesture}"
                    color = (0, 165, 255) # orange color

                case 'in-transaction':
                    frame, model_inference_time = YOLO_MODEL.inference(frame, time_now, MAIN_APP, AUDIO_HANDLER)
                    status_text = "YOLO Model: ACTIVE"
                    color = (0, 255, 0) # green color  

                case 'post-transaction':    
                    frame, current_gesture = GESTURE_MODEL.inference(frame, MAIN_APP, AUDIO_HANDLER)
                    status_text = f"Hand Gesture: {current_gesture}"
                    color = (0, 165, 255) # orange color

                case 'checkout':
                    status_text = "Checkout, please proceed to payment."
                    color = (255, 0, 0) # blue color  

                case _:
                    raise ValueError("Invalid app state")       

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
            elapsed = time_now - fps_time

            # Update FPS every 0.5 second
            if elapsed >= 0.5:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time_now
                model_fps = 1.0 / model_inference_time if model_inference_time > 0 else 0

            # Draw Model FPS
            cv2.putText(
                frame,
                f"Model FPS: {model_fps:.1f}",
                (frame.shape[1] - 250, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
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
                0.4,
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

@app.post("/app_state")
def get_app_state():
    global MAIN_APP
    return JSONResponse({"state": MAIN_APP.get_app_state()})

@app.get("/pos_items")
def get_pos_items():
    global MAIN_APP

    new_item_detected = MAIN_APP.new_item_added()
    if new_item_detected:
        MAIN_APP.last_count = MAIN_APP.total_qty

    return JSONResponse({"items": MAIN_APP.pos_items, "grand_total": MAIN_APP.grand_total, "new_item": new_item_detected})

# Start inference route
@app.get("/restart")
def restart_app():
    global MAIN_APP
    MAIN_APP.set_app_state('pre-transaction')
    MAIN_APP.clear_items()
    return {"status": "restarted"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/inference")
def inference_feed():
    return StreamingResponse(
        generate_inference_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )