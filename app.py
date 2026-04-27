import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import asyncio
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- CONFIGURATION ---
MODEL_DIR = "models"
MODELS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
CAM_WIDTH = 640
CAM_HEIGHT = 480
INFERENCE_EVERY_N_FRAMES = 3
JPEG_QUALITY = 70

# Your external inference server URL (set via env var)
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "")

# Drawing Connections
POSE_CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)]
HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

# --- Globals ---
executor = ThreadPoolExecutor(max_workers=3)
cap = None
pose_det = None
face_det = None
hand_det = None


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Done: {fname}")


def draw_landmarks(image, pose_res, face_res, hand_res):
    h, w = image.shape[:2]
    ann = image.copy()

    if pose_res and pose_res.pose_landmarks:
        for landmarks in pose_res.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(ann, pts[a], pts[b], (0, 220, 0), 2)
            for pt in pts:
                cv2.circle(ann, pt, 4, (0, 255, 0), -1)

    if face_res and face_res.face_landmarks:
        for landmarks in face_res.face_landmarks:
            for lm in landmarks:
                cv2.circle(ann, (int(lm.x * w), int(lm.y * h)), 1, (0, 165, 255), -1)

    if hand_res and hand_res.hand_landmarks:
        for landmarks in hand_res.hand_landmarks:
            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(hpts) and b < len(hpts):
                    cv2.line(ann, hpts[a], hpts[b], (60, 60, 255), 2)
            for pt in hpts:
                cv2.circle(ann, pt, 5, (80, 80, 255), -1)

    return ann


def landmarks_to_dict(pose_res, face_res, hand_res):
    """Serialize landmarks to a JSON-friendly dict for the inference server."""
    data = {"pose": [], "face": [], "hands": []}

    if pose_res and pose_res.pose_landmarks:
        for landmarks in pose_res.pose_landmarks:
            data["pose"].append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks])

    if face_res and face_res.face_landmarks:
        for landmarks in face_res.face_landmarks:
            data["face"].append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks])

    if hand_res and hand_res.hand_landmarks:
        for landmarks in hand_res.hand_landmarks:
            data["hands"].append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks])

    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap, pose_det, face_det, hand_det

    download_models()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    print(f"Camera opened: {cap.isOpened()}, index={CAM_INDEX}")
    yield

    cap.release()
    pose_det.close()
    face_det.close()
    hand_det.close()
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html")) as f:
        return f.read()


@app.websocket("/ws/video")
async def video_ws(ws: WebSocket):
    """Streams annotated MJPEG frames + landmark JSON to the browser."""
    await ws.accept()
    loop = asyncio.get_event_loop()

    frame_count = 0
    pose_res = face_res = hand_res = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            timestamp_ms = int(time.time() * 1000)

            if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
                fut_pose = executor.submit(pose_det.detect_for_video, mp_img, timestamp_ms)
                fut_face = executor.submit(face_det.detect_for_video, mp_img, timestamp_ms)
                fut_hand = executor.submit(hand_det.detect_for_video, mp_img, timestamp_ms)

                pose_res = await loop.run_in_executor(None, fut_pose.result)
                face_res = await loop.run_in_executor(None, fut_face.result)
                hand_res = await loop.run_in_executor(None, fut_hand.result)

            ann_frame = draw_landmarks(rgb_frame, pose_res, face_res, hand_res)

            # Encode as JPEG for efficient transport
            _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(ann_frame, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # Send frame as binary, landmarks as text JSON
            await ws.send_bytes(jpeg.tobytes())

            if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                landmark_data = landmarks_to_dict(pose_res, face_res, hand_res)
                await ws.send_text(json.dumps({"type": "landmarks", "data": landmark_data}))

            # Yield control so other connections can proceed
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/translate")
async def translate_ws(ws: WebSocket):
    """
    Bidirectional channel for your inference server integration.

    Browser sends: landmark JSON batches or control messages.
    Server responds: translated text / predictions.

    Wire this up to your inference server via httpx, aiohttp, or another WS client.
    """
    await ws.accept()
    try:
        while True:
            message = await ws.receive_text()
            payload = json.loads(message)

            # TODO: Forward payload to your inference server and return the result.
            # Example with httpx:
            #   async with httpx.AsyncClient() as client:
            #       resp = await client.post(INFERENCE_SERVER_URL, json=payload)
            #       result = resp.json()
            #       await ws.send_text(json.dumps(result))

            # Placeholder echo response
            await ws.send_text(json.dumps({
                "type": "translation",
                "text": "[connect your inference server]",
            }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
