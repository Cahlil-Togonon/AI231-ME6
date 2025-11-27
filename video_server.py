import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=640,height=480,framerate=30/1 ! "
    "jpegdec ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def mjpeg_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("error")
            continue
        
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            print("not working 2")
            continue

        frame_bytes = jpeg.tobytes()
        yield( b"--frame\r\n"
                b"Content-Type:image/jpeg\r\n"
                b"Content-Length: " +str(len(frame_bytes)).encode()+b"\r\n\r\n" + frame_bytes + b"\r\n")
            
@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boudary=frame")


@app.get("/")
def root():
    return {"message: Go to /video to see the stream"}