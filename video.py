import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

import mediapipe as mp
import cv2
import time

import threading
from queue import Queue
import pyttsx3

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

speech_queue = Queue()

def tts_worker():
    engine = pyttsx3.init('sapi5')   # force Windows engine
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 1.0)

    while True:
        text = speech_queue.get()

        try:
            print("🗣 SPEAK:", text)
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print("❌ TTS error:", e)

        speech_queue.task_done()

# Start exactly once
threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    if not text:
        return

    speech_queue.put_nowait(str(text))
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

# model = YOLO(f"../AI231/AI231_dataset/runs/{model_name}-{dataset_name}-augmented/weights/best.{format_extension[format]}")
model = YOLO(f"./models/best.{format_extension[format]}")
print(f"Loaded {model_name} model in {format} format")


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
    8:  {"name": "nestleyogurt", "price": 75},
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
    25: {"name": "Rambutan", "price": 15},
    26: {"name": "head&shoulders_shampoo", "price": 110},
    27: {"name": "irish-spring-soap", "price": 130},
    28: {"name": "c2_na_green", "price": 35},
    29: {"name": "colgate_toothpaste", "price": 150},
    30: {"name": "555-sardines-tomato", "price": 35},
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


# -------------------------------
# Mediapipe Hands for gesture recognition
# -------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
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
    global running, pos_items, last_OD_time, last_gesture_time, OD_running, GESTURE_running, COOLDOWN_PERIOD, GESTURE_RESTART
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

            now = time.time()

            if OD_running:
                results = model(frame, verbose=False)
                annotated = results[0].plot()
                frame = annotated

                if last_OD_time and now - last_OD_time > GESTURE_RESTART:
                    GESTURE_running = True

                if now - last_OD_time >= COOLDOWN_PERIOD:

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
            
            if GESTURE_running:
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
                            speak("Starting inference")
                        elif is_closed_fist(lm):
                            print("✊ Closed fist detected → Stopping inference...")
                            OD_running = False
                            speak("Stopping inference")


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
#   ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!doctype html>
    <html>

    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>Camera POS</title>
      <style>
        body { display:flex; margin:0; padding:0; height:100vh; background:black; color:white; }
        #cam { flex: 0 0 60%; object-fit: cover; width: 100%; height: 100%; }
        #pos { flex: 0 0 40%; padding: 10px; overflow: auto; background: #111; }
        table { width:100%; border-collapse: collapse; }
        th, td { border:1px solid #555; padding:5px; text-align:left; }
        button { padding:10px 20px; font-size:16px; margin-bottom:10px; }
      </style>
    </head>

    <body>
      <div>
        <img id="cam" src="/inference" alt="Camera stream">
      </div>
      <div id="pos">
        <button onclick="startInference()">Start</button>
        <h2>Items:</h2>
        <table id="pos_table">
          <tr><th>Qty</th><th>Name</th><th>Unit Price (₱)</th><th>Total Price (₱)</th></tr>
        </table>
        <h3>Grand Total: ₱<span id="grand_total">0</span></h3>
      </div>

      <audio id="beep_sound" src="/static/beep.mp3"></audio>

      <script>
        let beep_unlocked = false;

        function startInference() {
          fetch('/start');
        }

        // Unlock beep sound on first user interaction
        if (!beep_unlocked) {
        const beep = document.getElementById('beep_sound');
        beep.play().catch(() => {});  // play once to unlock
        beep.pause();
        beep.currentTime = 0;
        beep_unlocked = true;
        }

        async function updatePOS() {
          const response = await fetch('/pos_items');
          const data = await response.json();
          const table = document.getElementById('pos_table');

          // Clear table except header
          table.innerHTML = '<tr><th>Qty</th><th>Name</th><th>Unit Price (₱)</th><th>Total Price (₱)</th></tr>';

          // Add items
          for (const item of data.items) {
            const row = table.insertRow();
            row.insertCell(0).textContent = item.qty;
            row.insertCell(1).textContent = item.name;
            row.insertCell(2).textContent = item.unit_price;
            row.insertCell(3).textContent = item.total_price;
          }

          // Update grand total
          document.getElementById('grand_total').textContent = data.grand_total;
          
            // Play beep if new item detected
          if (data.new_item && beep_unlocked) {
            document.getElementById('beep_sound').play();
          }
        }

        setInterval(updatePOS, 500);
      </script>
    </body>
    </html>
    """
    return html

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

@app.get("/inference")
def inference_feed():
    return StreamingResponse(
        generate_inference_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
