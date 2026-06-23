import os
import time
import cv2

# 1. Set OpenCV FFMPEG transport options to TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;recv_buffer_size;10485760"

# 2. Configure video parameters
rtsp_url = "rtsp://admin:Kiran%4011@192.168.1.64:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
print(f"[INFO] Connecting to camera stream via OpenCV: {rtsp_url}")

# 3. Initialize OpenCV Capture using FFMPEG backend
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("[ERROR] Failed to open the camera stream.")
    exit(1)

print("[OK] Camera connection succeeded! Reading 15 frames...")
success_count = 0

# 4. Read 15 frames to verify the stream is reading successfully
for i in range(15):
    ret, frame = cap.read()
    if ret:
        success_count += 1
        print(f"  [Frame {success_count}] Successfully grabbed frame! Shape: {frame.shape}")
    else:
        print("  Waiting for frame...")
    time.sleep(0.1)

# Cleanup
cap.release()
print(f"\nFinished! Grabbed {success_count} frames successfully.")
