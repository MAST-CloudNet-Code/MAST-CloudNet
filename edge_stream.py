import cv2
import time
import requests
import argparse
import signal
import sys
import base64
import json
import socket
import subprocess
import os
from threading import Thread, Event
from queue import Queue, Full
import logging

# Try importing picamera2 (Raspberry Pi only)
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pi_video_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global settings
streaming = True
frame_queue = Queue(maxsize=10)  # Buffer a few frames
stop_event = Event()


class VideoStream:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.cap = None
        self.is_running = False
        self.width = 0
        self.height = 0
        self.fps = 0

    def start(self):
        if self.is_running:
            return True
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0:
                self.fps = 30
            logger.info(f"Started video capture: {self.width}x{self.height} @ {self.fps} FPS")
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False

    def read(self):
        if not self.is_running or self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Stopped video capture")


class PiCamera2Stream:
    def __init__(self, size=(1920, 1920), fps=1):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.is_running = True

    def start(self):
        return True  # Already started

    def read(self):
        if not self.is_running:
            return False, None
        frame = self.picam2.capture_array()
        return True, frame

    def stop(self):
        if self.is_running:
            self.picam2.stop()
            self.is_running = False
            logger.info("Stopped PiCamera2 stream")


class ImageStream:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.is_running = False
        self.width = 0
        self.height = 0

    def start(self):
        if self.is_running:
            return True
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                logger.error(f"Failed to load image from file: {self.image_path}")
                return False
            self.height, self.width = self.image.shape[:2]
            logger.info(f"Started image stream from file: {self.width}x{self.height}")
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Error loading image file: {e}")
            return False

    def read(self):
        if not self.is_running or self.image is None:
            return False, None
        return True, self.image.copy()

    def stop(self):
        self.image = None
        self.is_running = False
        logger.info("Stopped image stream")


def check_server_connectivity(server_url):
    try:
        if server_url.startswith("http://"):
            parts = server_url.replace("http://", "").split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0]) if len(parts) > 1 else 80
        elif server_url.startswith("https://"):
            parts = server_url.replace("https://", "").split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0]) if len(parts) > 1 else 443
        else:
            host = server_url
            port = 80
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((host, port))
        s.close()
        logger.info(f"Successfully connected to {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        return False


def capture_frames(stream, fps_limit):
    frame_delay = 1.0 / fps_limit
    frames_captured = 0
    start_time = time.time()
    while not stop_event.is_set():
        loop_start = time.time()
        ret, frame = stream.read()
        if not ret:
            logger.error("Failed to read frame from source")
            time.sleep(1)
            continue
        try:
            frame_queue.put_nowait(frame)
            frames_captured += 1
        except Full:
            pass
        if frames_captured % 100 == 0:
            elapsed = time.time() - start_time
            fps = frames_captured / elapsed if elapsed > 0 else 0
            logger.info(f"Captured {frames_captured} frames ({fps:.1f} FPS)")
        elapsed = time.time() - loop_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)


def send_frames(server_url, reconnect_interval=5):
    frames_sent = 0
    failed_attempts = 0
    last_success_time = None
    server_base_url = server_url.rstrip('/')
    receive_endpoint = f"{server_base_url}/receive_frame"
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except:
            continue
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            payload = {
                "frame": jpg_as_text,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "timestamp": time.time()
            }
            response = requests.post(receive_endpoint, json=payload, timeout=2)
            if response.status_code == 200:
                frames_sent += 1
                failed_attempts = 0
                last_success_time = time.time()
                if frames_sent % 50 == 0:
                    logger.info(f"Successfully sent {frames_sent} frames to server")
            else:
                logger.warning(f"Server returned error: {response.status_code}")
                failed_attempts += 1
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            failed_attempts += 1
        time.sleep(0.01)


def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down...")
    stop_event.set()
    time.sleep(1)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Stream video from Raspberry Pi to server")
    parser.add_argument("--camera", type=str, default="0",
                        help="Camera index (0,1...) or /dev/video path")
    parser.add_argument("--image", type=str, default=None,
                        help="Image file to stream instead of camera")
    parser.add_argument("--server", type=str, default="http://localhost:5000",
                        help="Server URL (e.g., http://192.168.0.100:5000)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Max FPS (default 15)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    if args.image is not None:
        if not os.path.isfile(args.image):
            logger.error(f"Image file not found: {args.image}")
            return
        logger.info(f"Using image file: {args.image}")
        stream = ImageStream(args.image)
    else:
        if HAS_PICAMERA2:
            logger.info("Using PiCamera2 backend")
            stream = PiCamera2Stream(size=(1920, 1920), fps=args.fps)
        else:
            try:
                camera_source = int(args.camera)
            except ValueError:
                camera_source = args.camera
            logger.info(f"Using OpenCV VideoCapture source: {camera_source}")
            stream = VideoStream(camera_source)

    if not stream.start():
        logger.error("Failed to start stream, exiting")
        return

    if not check_server_connectivity(args.server):
        logger.warning("Initial connection to server failed. Will retry later.")

    capture_thread = Thread(target=capture_frames, args=(stream, args.fps))
    capture_thread.daemon = True
    capture_thread.start()

    sender_thread = Thread(target=send_frames, args=(args.server,))
    sender_thread.daemon = True
    sender_thread.start()

    logger.info(f"Started streaming to {args.server}")
    logger.info("Press Ctrl+C to stop")
    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        stop_event.set()
        stream.stop()
        logger.info("Stream ended")


if __name__ == "__main__":
    main()
