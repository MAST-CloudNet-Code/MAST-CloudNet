from flask import Flask, render_template, jsonify, url_for, request, send_file, Response
import cv2
import threading
import time
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
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

EXPERIMENT_LOG_FILE = 'experiment_log.txt'
PERFORMANCE_LOG_FILE = 'server_performance_log.csv'


class AsyncCSVLogger:
    def __init__(self, filename, headers):
        self.filename = filename
        self.queue = queue.Queue()
        self.headers = headers
        self.stop_logging = False

        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def log(self, data):
        self.queue.put(data)

    def _worker(self):
        while not self.stop_logging:
            try:
                data = self.queue.get(timeout=0.5)
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Logger Error: {e}")


perf_logger = AsyncCSVLogger(
    PERFORMANCE_LOG_FILE,
    [
        'Log_Time', 'Frame_ID',
        'Edge_Capture_TS', 'Server_Recv_TS', 'Proc_Start_TS', 'Proc_End_TS',
        'Network_Latency_ms', 'Queue_Wait_ms', 'Inference_Time_ms', 'Total_Latency_ms',
        'Detections'
    ]
)


class LastFrameQueue:
    def __init__(self, maxsize=1):
        self.q = queue.Queue(maxsize=maxsize)

    def put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except queue.Empty:
                pass


try:
    model = YOLO('models/best.pt')
    model.model.names = {0: 'Aedes', 1: 'Non-Aedes'}
    logger.info("Successfully loaded YOLO model")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

CLASSES_OF_INTEREST = ['Aedes', 'Non-Aedes']

lock = threading.Lock()
stop_processing = threading.Event()

latest_count = {cls: 0 for cls in CLASSES_OF_INTEREST}
unique_ids_per_class = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
logged_track_ids = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
detection_log = []

frame_queue = LastFrameQueue(maxsize=2)
frame_buffer = deque(maxlen=1)

CONFIDENCE_THRESHOLD = 0.6
STATUS_TIMEOUT = 5
last_frame_time = None

frame_stats = {
    "total_received": 0,
    "processing_fps": 0,
    "last_latency_ms": 0,
    "network_latency_ms": 0
}


def signal_handler(sig, frame):
    logger.info('Signal received, exiting...')
    stop_processing.set()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def process_frame(frame, metadata):
    global latest_count, unique_ids_per_class, detection_log

    if model is None: return frame, 0

    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detections_count = 0

    try:
        results = model.track(
            source=frame,
            tracker='bytetrack.yaml',
            persist=True,
            imgsz=1920,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        if results and results[0].boxes:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()

            detections_count = len(boxes)

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy()
            else:
                track_ids = [None] * len(boxes)

            for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                cls_name = model.names[int(cls_id)]

                if cls_name in CLASSES_OF_INTEREST:
                    x1, y1, x2, y2 = map(int, box)

                    color = (255, 0, 0) if cls_name == 'Aedes' else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, f"{cls_name} {track_id if track_id else ''}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    if track_id is not None:
                        with lock:
                            unique_ids_per_class[cls_name].add(track_id)

                            if track_id not in logged_track_ids[cls_name]:
                                logged_track_ids[cls_name].add(track_id)
                                detection_log.append({
                                    "timestamp": current_time_str,
                                    "class": cls_name,
                                    "track_id": int(track_id)
                                })
                                with open(EXPERIMENT_LOG_FILE, 'a') as f:
                                    f.write(f"{current_time_str} | {cls_name} | ID: {track_id}\n")

        with lock:
            latest_count = {cls: len(unique_ids_per_class[cls]) for cls in CLASSES_OF_INTEREST}

    except Exception as e:
        logger.error(f"YOLO Error: {e}")

    return frame, detections_count


def process_frames_thread():
    global last_frame_time, frame_stats
    logger.info("Started frame processing thread")

    while not stop_processing.is_set():
        try:
            item = frame_queue.q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame, meta = item

        proc_start = time.time()
        processed_frame, det_count = process_frame(frame, meta)
        proc_end = time.time()

        edge_ts = meta['edge_ts']
        recv_ts = meta['recv_ts']
        frame_id = meta['frame_id']

        network_lat = (recv_ts - edge_ts) * 1000
        queue_wait = (proc_start - recv_ts) * 1000
        inference_time = (proc_end - proc_start) * 1000
        total_lat = (proc_end - edge_ts) * 1000

        perf_logger.log([
            datetime.now().strftime('%H:%M:%S.%f'),
            frame_id,
            f"{edge_ts:.4f}",
            f"{recv_ts:.4f}",
            f"{proc_start:.4f}",
            f"{proc_end:.4f}",
            f"{network_lat:.2f}",
            f"{queue_wait:.2f}",
            f"{inference_time:.2f}",
            f"{total_lat:.2f}",
            det_count
        ])

        with lock:
            frame_buffer.append(processed_frame)
            last_frame_time = proc_end

            frame_stats["processing_fps"] = 1000.0 / inference_time if inference_time > 0 else 0
            frame_stats["last_latency_ms"] = total_lat
            frame_stats["network_latency_ms"] = network_lat


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})


@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    global frame_stats
    recv_time = time.time()

    try:
        try:
            client_timestamp = float(request.headers.get('X-Timestamp', recv_time))
            frame_id = request.headers.get('X-Frame-ID', 'unknown')
        except ValueError:
            client_timestamp = recv_time
            frame_id = 'error'

        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error"}), 400

        network_latency = (recv_time - client_timestamp) * 1000

        with lock:
            frame_stats["total_received"] += 1
            frame_stats["network_latency_ms"] = network_latency

        metadata = {
            'edge_ts': client_timestamp,
            'recv_ts': recv_time,
            'frame_id': frame_id
        }

        frame_queue.put((frame, metadata))
        return "OK", 200

    except Exception as e:
        logger.error(f"Receive error: {e}")
        return "Error", 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/frame')
def frame_feed():
    with lock:
        if not frame_buffer:
            return jsonify({"status": "no_frame"})
        out_frame = frame_buffer[-1]
        ret, buffer = cv2.imencode('.jpg', out_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/stats')
def stats():
    with lock:
        s = frame_stats.copy()
        s["counts"] = latest_count.copy()
        s["status"] = "online" if (last_frame_time and time.time() - last_frame_time < STATUS_TIMEOUT) else "offline"
        s["queue_size"] = frame_queue.q.qsize()
        return jsonify(s)


@app.route('/reset', methods=['POST'])
def reset():
    global latest_count, unique_ids_per_class, logged_track_ids, detection_log
    with lock:
        unique_ids_per_class = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
        logged_track_ids = {cls_name: set() for cls_name in CLASSES_OF_INTEREST}
        latest_count = {cls: 0 for cls in CLASSES_OF_INTEREST}
        detection_log.clear()

        with open(EXPERIMENT_LOG_FILE, 'w') as f: f.write("")
        with open(PERFORMANCE_LOG_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(perf_logger.headers)

    return jsonify({"status": "reset"})


@app.route('/export_counts')
def export_counts():
    with lock:
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Class', 'Total Unique Count'])
        for cls, count in latest_count.items():
            cw.writerow([cls, count])
        cw.writerow([])
        cw.writerow(['Timestamp', 'Class', 'Track ID'])
        for entry in detection_log:
            cw.writerow([entry.get('timestamp'), entry.get('class'), entry.get('track_id')])
        output = io.BytesIO()
        output.write(si.getvalue().encode('utf-8'))
        output.seek(0)

    return send_file(
        output, mimetype='text/csv', as_attachment=True,
        download_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route('/export_logs')
def export_logs():
    """Zips the performance log and experiment log for download"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        if os.path.exists(PERFORMANCE_LOG_FILE):
            zf.write(PERFORMANCE_LOG_FILE, arcname='system_performance.csv')

        if os.path.exists(EXPERIMENT_LOG_FILE):
            zf.write(EXPERIMENT_LOG_FILE, arcname='detection_log.txt')

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )



if not any(t.name == 'FrameProcessor' for t in threading.enumerate()):
    t = threading.Thread(target=process_frames_thread, daemon=True, name='FrameProcessor')
    t.start()
    logger.info("BACKGROUND THREAD STARTED VIA GUNICORN")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    try:
        from waitress import serve
        logger.info(f"Running on port {args.port}")
        serve(app, host='0.0.0.0', port=args.port, threads=6)
    except ImportError:
        app.run(host='0.0.0.0', port=args.port, threaded=True)