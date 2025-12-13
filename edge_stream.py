import cv2
import time
import requests
import argparse
import signal
import sys
import os
import csv
from threading import Thread, Event
from queue import Queue, Full, Empty
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

stop_event = Event()


# --- 1. ASYNC LOGGER (Efficient Protocol) ---
class AsyncCSVLogger:
    """
    Logs data to CSV in a separate thread to prevent blocking
    the main video capture or network threads.
    """

    def __init__(self, filename, headers):
        self.filename = filename
        self.queue = Queue()
        self.headers = headers
        self.stop_logging = False

        # Initialize file with headers
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def log(self, data):
        """Add data list to queue"""
        self.queue.put(data)

    def _worker(self):
        while not self.stop_logging or not self.queue.empty():
            try:
                # Batch write could be implemented here for even higher efficiency
                data = self.queue.get(timeout=0.5)
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Logging error: {e}")

    def stop(self):
        self.stop_logging = True
        self.thread.join()


# Initialize Logger
edge_logger = AsyncCSVLogger(
    'edge_performance_log.csv',
    ['Timestamp', 'Frame_ID', 'Capture_Time', 'Send_Start', 'Send_End', 'Tx_Duration', 'Queue_Size']
)


# --- Optimized LIFO Queue ---
class LIFOQueue(Queue):
    """Queue that drops the oldest item when full"""

    def put(self, item, block=True, timeout=None):
        if self.full():
            try:
                self.get_nowait()
            except Empty:
                pass
        super().put(item, block, timeout)


# Queue stores: (frame_data, capture_timestamp, frame_id)
frame_queue = LIFOQueue(maxsize=2)


class VideoStream:
    def __init__(self, src=0, use_picamera=False, size=(1920, 1920), fps=5):
        self.src = src
        self.use_picamera = use_picamera
        self.size = size
        self.target_fps = fps
        self.cap = None
        self.picam2 = None

    def start(self):
        if self.use_picamera:
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": self.size, "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                logger.info("PiCamera2 started")
            except ImportError:
                logger.error("PiCamera2 not found, falling back to OpenCV")
                self.use_picamera = False

        if not self.use_picamera:
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            logger.info(f"OpenCV Capture started on {self.src}")

    def read(self):
        if self.use_picamera:
            return True, self.picam2.capture_array()
        else:
            if self.cap and self.cap.isOpened():
                return self.cap.read()
            return False, None

    def release(self):
        if self.picam2: self.picam2.stop()
        if self.cap: self.cap.release()


class ImageStream:
    def __init__(self, image_path):
        self.image_path = image_path
        self.frame = None

    def start(self):
        if not os.path.exists(self.image_path):
            logger.error(f"Image file not found: {self.image_path}")
            sys.exit(1)
        self.frame = cv2.imread(self.image_path)
        if self.frame is None:
            logger.error("Failed to read image file")
            sys.exit(1)
        logger.info(f"Loaded image: {self.image_path}")

    def read(self):
        if self.frame is not None:
            return True, self.frame.copy()
        return False, None

    def release(self):
        pass


def capture_worker(stream, fps_limit):
    """Captures frames, assigns ID, and tags with EXACT capture time"""
    delay = 1.0 / fps_limit
    frame_id_counter = 0

    while not stop_event.is_set():
        start_time = time.time()

        ret, frame = stream.read()
        if ret and frame is not None:
            capture_time = time.time()
            frame_id_counter += 1

            # Put Tuple: (Frame, Timestamp, ID)
            frame_queue.put((frame, capture_time, frame_id_counter))
        else:
            time.sleep(0.1)
            continue

        elapsed = time.time() - start_time
        if elapsed < delay:
            time.sleep(delay - elapsed)


def sender_worker(server_url):
    """Encodes and sends frames using the capture timestamp and ID"""
    endpoint = f"{server_url}/receive_frame"
    session = requests.Session()

    logger.info(f"Streaming to {endpoint}")

    while not stop_event.is_set():
        try:
            # Retrieve: (frame, timestamp, id)
            item = frame_queue.get(timeout=1.0)
            frame, capture_ts, frame_id = item

            send_start = time.time()

            # Encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)

            if not ret: continue

            data_bytes = buffer.tobytes()

            # --- Protocol: Send ID and TS in Headers ---
            headers = {
                'Content-Type': 'application/octet-stream',
                'X-Timestamp': str(capture_ts),
                'X-Frame-ID': str(frame_id)
            }

            response = session.post(
                endpoint,
                data=data_bytes,
                headers=headers,
                timeout=10.0
            )

            send_end = time.time()

            # --- Log Performance ---
            edge_logger.log([
                datetime.now().strftime('%H:%M:%S.%f'),
                frame_id,
                f"{capture_ts:.4f}",
                f"{send_start:.4f}",
                f"{send_end:.4f}",
                f"{(send_end - send_start) * 1000:.2f}",  # Tx Duration ms
                frame_queue.qsize()
            ])

            if response.status_code != 200:
                logger.warning(f"Server Rejected: {response.status_code}")

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Connection Error: {e}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:5000")
    parser.add_argument("--camera", default=0)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--picam", action="store_true")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())

    if args.image:
        stream = ImageStream(args.image)
    else:
        cam_src = int(args.camera) if str(args.camera).isdigit() else args.camera
        stream = VideoStream(src=cam_src, use_picamera=args.picam, fps=args.fps)

    stream.start()

    t_cap = Thread(target=capture_worker, args=(stream, args.fps), daemon=True)
    t_send = Thread(target=sender_worker, args=(args.server,), daemon=True)

    t_cap.start()
    t_send.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        edge_logger.stop()
        stream.release()
        logger.info("Exiting")


if __name__ == "__main__":
    main()