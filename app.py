from flask import Flask, render_template, jsonify, url_for, request, send_file
import cv2
import threading
import time
import base64
from ultralytics import YOLO
import signal
import sys
from collections import deque
import io
import csv
from datetime import datetime
import os
import argparse
import numpy as np
import logging
import queue


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

EXPERIMENT_LOG_FILE = 'experiment_log.txt'


if not os.path.isfile(EXPERIMENT_LOG_FILE):
    with open(EXPERIMENT_LOG_FILE, 'w') as f:
        f.write("Experiment Log Started\n\n")


try:
    model = YOLO('models/best.pt')
    model.model.names = {0: 'Aedes', 1: 'Non-Aedes'}
    logger.info("Successfully loaded YOLO model")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

CLASSES_OF_INTEREST = ['Aedes', 'Non-Aedes']

latest_frame = None
latest_count = {cls: 0 for cls in CLASSES_OF_INTEREST}
lock = threading.Lock()

frame_queue = queue.Queue(maxsize=30)
frame_buffer = deque(maxlen=15)

NMS_THRESHOLD = 1
CONFIDENCE_THRESHOLD = 0.6
MAX_WIDTH = 2000
MAX_HEIGHT = 2000

last_frame_time = None
STATUS_TIMEOUT = 5


detection_log = []

unique_ids_per_class = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
logged_track_ids = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}

frame_stats = {
    "total_received": 0,
    "last_received_time": None,
    "processing_fps": 0,
    "queue_size": 0
}

stop_processing = threading.Event()


def is_class_of_interest(cls_name):
    return cls_name in CLASSES_OF_INTEREST


def signal_handler(sig, frame):
    logger.info('Signal received, exiting...')
    stop_processing.set()
    time.sleep(1)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def process_frame(frame):
    """Process a single frame with YOLO model"""
    global latest_count, unique_ids_per_class, detection_log, logged_track_ids

    if model is None:
        logger.warning("YOLO model not loaded, cannot process frame")
        return frame

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    counts = {cls_name: 0 for cls_name in CLASSES_OF_INTEREST}

    try:
        results = model.track(
            source=frame,
            tracker='bytetrack.yaml',
            persist=True,
            imgsz=(1920, 1920),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]

            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                try:
                    if (hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None and
                            hasattr(result.boxes, 'conf') and result.boxes.conf is not None and
                            hasattr(result.boxes, 'cls') and result.boxes.cls is not None):

                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy()

                        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                            track_ids = result.boxes.id.cpu().numpy()
                        else:
                            track_ids = [None] * len(boxes)

                        rects = []
                        confidences = []
                        class_ids = []
                        track_ids_list = []

                        for box, cls_id, conf, track_id in zip(boxes, cls_ids, confs, track_ids):
                            if conf < CONFIDENCE_THRESHOLD:
                                continue

                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1

                            if width > MAX_WIDTH or height > MAX_HEIGHT:
                                continue

                            rects.append([int(x1), int(y1), int(width), int(height)])
                            confidences.append(float(conf))
                            class_ids.append(int(cls_id))
                            track_ids_list.append(track_id)

                        indices = cv2.dnn.NMSBoxes(rects, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

                        for i in indices.flatten():
                            x, y, w, h = rects[i]
                            conf = confidences[i]
                            cls_id = class_ids[i]
                            track_id = track_ids_list[i]

                            cls_name = model.names[cls_id]
                            if is_class_of_interest(cls_name):
                                counts[cls_name] += 1
                                if track_id is not None:
                                    unique_ids_per_class[cls_name].add(track_id)

                                    if cls_name == 'Aedes':
                                        box_color = (255, 0, 0)
                                        label = f"Aedes {conf:.2f}"
                                        text_color = (255, 255, 255)
                                    else:
                                        box_color = (255, 255, 0)
                                        label = f"Non Aedes {conf:.2f}"
                                        text_color = (0, 0, 0)

                                    (font_width, font_height), baseline = cv2.getTextSize(label,
                                                                                          cv2.FONT_HERSHEY_SIMPLEX,
                                                                                          2,
                                                                                          8)


                                    label_background = (x, y - font_height - 10, x + font_width, y)


                                    cv2.rectangle(frame, (label_background[0], label_background[1]),
                                                  (label_background[2], label_background[3]), box_color,
                                                  -1)


                                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color,
                                                  4)


                                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color,
                                                8, cv2.LINE_AA)


                                    with lock:
                                        if track_id not in logged_track_ids[cls_name]:
                                            detection_log.append({
                                                "timestamp": current_time,
                                                "class": cls_name,
                                                "track_id": track_id,
                                                "confidence": conf
                                            })
                                            logged_track_ids[cls_name].add(track_id)
                                            log_line = f"{current_time} | Class: {cls_name} | Track ID: {track_id} | Confidence: {conf:.2f}\n"
                                            with open(EXPERIMENT_LOG_FILE, 'a') as f:
                                                f.write(log_line)

                except Exception as e:
                    logger.error(f"Error processing detection boxes: {e}")

        with lock:

            latest_count = {cls: len(unique_ids_per_class[cls]) for cls in CLASSES_OF_INTEREST}

    except Exception as e:
        logger.error(f"Error in YOLO processing: {e}")

    return frame


def process_frames_thread():
    """Thread to process frames in the background"""
    global last_frame_time, frame_stats

    logger.info("Started frame processing thread")

    while not stop_processing.is_set():
        try:

            try:
                frame_to_process = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            process_start = time.time()


            processed_frame = process_frame(frame_to_process)

            with lock:

                frame_buffer.append(processed_frame)


                process_time = time.time() - process_start
                frame_stats["processing_fps"] = 1.0 / process_time if process_time > 0 else 0
                frame_stats["queue_size"] = frame_queue.qsize()
                last_frame_time = time.time()


            frame_queue.task_done()

        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            time.sleep(1)


@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to check if server is alive"""
    return jsonify({"status": "ok", "server_time": datetime.now().isoformat()})


@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """Endpoint to receive frames from the Raspberry Pi"""
    global last_frame_time, frame_stats

    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"status": "error", "message": "No frame data received"}), 400


        jpg_as_text = data['frame']
        jpg_original = base64.b64decode(jpg_as_text)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Could not decode frame"}), 400

        with lock:
            last_frame_time = time.time()
            frame_stats["total_received"] += 1
            frame_stats["last_received_time"] = time.time()

        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            logger.warning("Frame queue is full, dropping frame")

        return jsonify({"status": "ok"})

    except Exception as e:
        logger.error(f"Error receiving frame: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html', count_url=url_for('count'))


@app.route('/frame')
def frame():
    with lock:
        if not frame_buffer:
            return jsonify({"frame": None})
        frame = frame_buffer[-1]

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({"frame": None})
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        height, width, _ = frame.shape

    return jsonify({"frame": jpg_as_text, "width": width, "height": height})


@app.route('/count')
def count():
    with lock:
        return jsonify(latest_count.copy())


@app.route('/stats')
def stats():
    with lock:
        current_stats = frame_stats.copy()
        current_stats["camera_status"] = "online" if (last_frame_time is not None and
                                                      time.time() - last_frame_time < STATUS_TIMEOUT) else "offline"
        current_stats["counts"] = latest_count.copy()
        return jsonify(current_stats)


@app.route('/reset_count', methods=['POST'])
def reset_count():
    global unique_ids_per_class, latest_count, detection_log, logged_track_ids
    with lock:
        unique_ids_per_class = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
        latest_count = {cls: 0 for cls in CLASSES_OF_INTEREST}
        detection_log = []
        logged_track_ids = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
    return jsonify({"status": "success", "message": "Counts reset"})


@app.route('/status')
def status():
    with lock:
        if last_frame_time is None:
            return jsonify({"status": "offline"})
        if time.time() - last_frame_time < STATUS_TIMEOUT:
            return jsonify({"status": "online"})
        else:
            return jsonify({"status": "offline"})


@app.route('/export_counts')
def export_counts():
    with lock:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        writer.writerow(['Class', 'Unique Count'])
        for cls, count in latest_count.items():
            writer.writerow([cls, count])

        writer.writerow([])
        writer.writerow(['Detection Log'])
        writer.writerow(['Timestamp', 'Class', 'Track ID', 'Confidence'])

        for entry in detection_log:
            writer.writerow([entry['timestamp'], entry['class'], entry['track_id'], f"{entry['confidence']:.2f}"])

        csv_buffer.seek(0)

        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='mosquito_detection_log.csv'
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Mosquito Detection Server")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the web server on")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads to process frames")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for _ in range(max(1, args.workers)):
        t = threading.Thread(target=process_frames_thread, daemon=True)
        t.start()

    logger.info(f"Starting web server on port {args.port} with {args.workers} worker threads")

    try:
        from waitress import serve

        logger.info("Using Waitress production server")
        serve(app, host='0.0.0.0', port=args.port, threads=16)
    except ImportError:
        logger.warning("Waitress not available, using Flask's development server")
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)