import cv2
import time
import requests
import argparse
import signal
import sys
import base64
import socket
import subprocess
import os
from threading import Thread, Event
from queue import Queue, Full
import logging

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
        """Initialize video stream from camera"""
        self.camera_source = camera_source
        self.cap = None
        self.is_running = False
        self.width = 0
        self.height = 0
        self.fps = 0

    def start(self):
        """Start video capture"""
        if self.is_running:
            return True

        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False

            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0:
                self.fps = 30  # Default FPS

            logger.info(f"Started video capture: {self.width}x{self.height} @ {self.fps} FPS")
            self.is_running = True
            return True

        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False

    def read(self):
        """Read a frame from the video source"""
        if not self.is_running or self.cap is None:
            return False, None

        return self.cap.read()

    def stop(self):
        """Stop video capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Stopped video capture")


class ImageStream:
    def __init__(self, image_path):
        """Initialize stream from a static image file, for testing purposes"""
        self.image_path = image_path
        self.image = None
        self.is_running = False
        self.width = 0
        self.height = 0

    def start(self):
        """Load the image file"""
        if self.is_running:
            return True

        try:
            # Load the image file
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                logger.error(f"Failed to load image from file: {self.image_path}")
                return False

            # Get image properties
            self.height, self.width = self.image.shape[:2]
            logger.info(f"Started image stream from file: {self.width}x{self.height}")
            self.is_running = True
            return True

        except Exception as e:
            logger.error(f"Error loading image file: {e}")
            return False

    def read(self):
        """Return the loaded image"""
        if not self.is_running or self.image is None:
            return False, None

        # Always return the same image
        return True, self.image.copy()

    def stop(self):
        """Stop image streaming"""
        self.image = None
        self.is_running = False
        logger.info("Stopped image stream")


def check_server_connectivity(server_url):
    """Check if the server is reachable"""
    try:
        # Extract host and port from URL
        # default ports: 80 for http, 443 for https
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

        # Try to connect to the server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((host, port))
        s.close()

        # If we get here, the connection was successful
        logger.info(f"Successfully connected to {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")

        # Try ping as a fallback
        try:
            ping_result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if ping_result.returncode == 0:
                logger.info(f"Ping to {host} successful, but port {port} may be closed")
            else:
                logger.error(f"Ping to {host} failed")
        except Exception as ping_err:
            logger.error(f"Ping check failed: {ping_err}")

        return False


def capture_frames(stream, fps_limit):
    """Capture frames from camera and put them in the queue"""
    frame_delay = 1.0 / fps_limit
    frames_captured = 0
    start_time = time.time()

    while not stop_event.is_set():
        loop_start = time.time()

        # Capture frame
        ret, frame = stream.read()
        if not ret:
            logger.error("Failed to read frame from source")
            time.sleep(1)  # Wait before retrying
            continue

        # Resize frame if needed to reduce bandwidth
        # frame = cv2.resize(frame, (640, 480))

        # add to queue, non-blocking
        try:
            frame_queue.put_nowait(frame)
            frames_captured += 1
        except Full:
            # Queue is full, which means the sender can't keep up
            # Skip this frame
            pass

        # Log stats periodically
        if frames_captured % 100 == 0:
            elapsed = time.time() - start_time
            fps = frames_captured / elapsed if elapsed > 0 else 0
            logger.info(f"Captured {frames_captured} frames ({fps:.1f} FPS)")

        # Control frame rate
        elapsed = time.time() - loop_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)


def send_frames(server_url, reconnect_interval=5):
    """Send frames from queue to server"""
    frames_sent = 0
    failed_attempts = 0
    last_success_time = None

    server_base_url = server_url.rstrip('/')
    receive_endpoint = f"{server_base_url}/receive_frame" #defined by the webapp running on the server

    # Support for HTTPS
    verify_ssl = True
    if server_url.startswith("https://"):
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            # Set to False if using self-signed certificates
            # verify_ssl = False
        except ImportError:
            logger.warning("urllib3 not available, SSL warnings will be shown")

    while not stop_event.is_set():
        # Check if we need to test connectivity after multiple failures
        if failed_attempts >= 5:
            logger.warning(f"Multiple failures ({failed_attempts}), checking server connectivity...")
            if not check_server_connectivity(server_url):
                logger.error(f"Server unreachable, waiting {reconnect_interval} seconds before retry")
                time.sleep(reconnect_interval)
                continue
            failed_attempts = 0

        # Try to get a frame from the queue
        try:
            frame = frame_queue.get(timeout=1.0)
        except:
            # No frame available
            continue

        # Convert to JPEG
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Prepare payload
            payload = {
                "frame": jpg_as_text,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "timestamp": time.time()
            }

            # Send to server
            response = requests.post(
                receive_endpoint,
                json=payload,
                timeout=2,  # Short timeout
                verify=verify_ssl  # SSL verification based on URL
            )

            if response.status_code == 200:
                frames_sent += 1
                failed_attempts = 0
                last_success_time = time.time()

                # Log progress periodically
                if frames_sent % 50 == 0:
                    logger.info(f"Successfully sent {frames_sent} frames to server")

            else:
                logger.warning(f"Server returned error: {response.status_code} - {response.text}")
                failed_attempts += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending frame: {e}")
            failed_attempts += 1

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            failed_attempts += 1

        # Check if we've had recent success
        if last_success_time is not None and time.time() - last_success_time > 60:
            logger.warning("No successful transmissions for 60 seconds")

        # Brief pause to prevent hammering the server
        time.sleep(0.01)


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Interrupt received, shutting down...")
    stop_event.set()
    time.sleep(1)  # Give threads time to finish
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Send Frames from Raspberry Pi to server")
    parser.add_argument("--camera", type=str, default="0",
                        help="Camera index (0, 1...) or video file path")
    parser.add_argument("--image", type=str, default=None,
                        help="Image file instead of camera for testing")
    parser.add_argument("--server", type=str, default="http://localhost:5000",
                        help="Server URL")
    parser.add_argument("--fps", type=int, default=15,
                        help="Maximum FPS to capture (default: 15)")
    parser.add_argument("--retry", type=int, default=5,
                        help="timeout (default: 5s)")
    parser.add_argument("--no-verify-ssl", action="store_true",
                        help="Disable SSL certificate verification for HTTPS connections")

    args = parser.parse_args()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Handle SSL verification setting
    if args.no_verify_ssl and args.server.startswith("https://"):
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.info("SSL certificate verification disabled")

    # Determine the stream source
    if args.image is not None:
        # Use image file
        if not os.path.isfile(args.image):
            logger.error(f"Image file not found: {args.image}")
            return

        logger.info(f"Using image file: {args.image}")
        stream = ImageStream(args.image)
    else:
        # Use camera
        # If camera is numeric, convert to int
        try:
            camera_source = int(args.camera)
        except ValueError:
            # Keep as string if it's a file path
            camera_source = args.camera

        logger.info(f"Using camera source: {camera_source}")
        stream = VideoStream(camera_source)

    # Start the stream
    if not stream.start():
        logger.error("Failed to start stream, exiting")
        return

    # Check initial connectivity
    if not check_server_connectivity(args.server):
        logger.warning(f"Initial connection to server failed. Will keep trying in background.")

    # Start capture thread
    capture_thread = Thread(target=capture_frames, args=(stream, args.fps))
    capture_thread.daemon = True
    capture_thread.start()

    # Start sender thread
    sender_thread = Thread(target=send_frames, args=(args.server, args.retry))
    sender_thread.daemon = True
    sender_thread.start()

    logger.info(f"Started streaming to {args.server}")
    logger.info("Press Ctrl+C to stop")

    try:
        # Keep main thread alive
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Clean up
        stop_event.set()
        stream.stop()
        logger.info("Stream ended")


if __name__ == "__main__":
    main()
