import cv2
import sys
import subprocess
import socket

def check_ping(ip):
    print(f"Pinging {ip}...")
    try:
        # On Windows, -n 2 sends 2 ping packets
        res = subprocess.run(["ping", "-n", "2", ip], capture_output=True, text=True, timeout=5)
        print(res.stdout)
        return res.returncode == 0
    except Exception as e:
        print(f"Ping failed to execute: {str(e)}")
        return False

def check_port(ip, port):
    print(f"Checking if port {port} is open on {ip}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3.0)
    try:
        s.connect((ip, port))
        print(f"  Port {port} is OPEN!")
        s.close()
        return True
    except Exception as e:
        print(f"  Port {port} is CLOSED or unreachable: {str(e)}")
        s.close()
        return False

def test_rtsp(url):
    print(f"Testing connection to: {url}")
    # Set lower timeout for ffmpeg to avoid waiting 30 seconds
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|timeout;5000000" # 5 sec timeout
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[-] Failed to open the camera stream.")
        return False
        
    print("[+] Successfully connected! Reading 5 frames...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1} read successfully, shape: {frame.shape}")
        else:
            print(f"  Frame {i+1} failed to read.")
            
    cap.release()
    return True

if __name__ == "__main__":
    ip = "192.168.1.64"
    
    # 1. Network Diagnostics
    ping_ok = check_ping(ip)
    port_rtsp_ok = check_port(ip, 554)
    port_http_ok = check_port(ip, 80)
    port_sdk_ok = check_port(ip, 8000) # Hikvision/Prama control port
    
    if not ping_ok:
        print("[!] Warning: Host is not responding to ping. Check if IP is correct and on the same subnet.")
    
    # 2. Try RTSP URLs if ports are open
    if port_rtsp_ok:
        urls = [
            "rtsp://admin:Kiran%4011@192.168.1.64:554/Streaming/Channels/101",
            "rtsp://admin:Kiran@11@192.168.1.64:554/Streaming/Channels/101",
            "rtsp://admin:Kiran%4011@192.168.1.64/Streaming/Channels/101",
            "rtsp://admin:Kiran%4011@192.168.1.64:554/h264/ch1/main/av_stream",
        ]
        
        for url in urls:
            print("\n" + "="*50)
            try:
                success = test_rtsp(url)
                if success:
                    print("[+] Found working URL!")
                    sys.exit(0)
            except Exception as e:
                print(f"[-] Exception occurred: {str(e)}")
    else:
        print("\n[!] RTSP port 554 is closed. You must enable RTSP/ONVIF in your camera configuration webpage.")
