import cv2
import numpy as np
import socket
import threading

# --- CONFIGURATION ---
ESP32_IP = "192.168.137.126"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
UDP_PORT = 4210

CONFIRM_ON_FRAMES = 5
CONFIRM_OFF_FRAMES = 10

# Create UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_signal_async(state):
    """Sends UDP packet."""
    def task():
        try:
            message = b"1" if state else b"0"
            udp_socket.sendto(message, (ESP32_IP, UDP_PORT))
            print(f">>> UDP Sent: {'ON' if state else 'OFF'}")
        except Exception as e:
            print(f"UDP Error: {e}")
    
    threading.Thread(target=task).start()

def run_detection():
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(STREAM_URL)
    
    is_person_present = False
    signal_sent = False
    on_counter = 0
    off_counter = 0

    print("System started. Looking for people...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_this_frame = False
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.5 and classes[class_id] == "person":
                    detected_this_frame = True
                    break

        if detected_this_frame:
            on_counter += 1
            off_counter = 0
        else:
            off_counter += 1
            on_counter = 0

        if not is_person_present and on_counter >= CONFIRM_ON_FRAMES:
            is_person_present = True
            if not signal_sent:
                #send_signal_async(1)
                signal_sent = True
                print("🔔 NEW PERSON DETECTED - Signal sent!")

        elif is_person_present and off_counter >= CONFIRM_OFF_FRAMES:
            #send_signal_async(0)
            is_person_present = False
            signal_sent = False
            print("👋 Person left - Ready for next detection")

        status_color = (0, 255, 0) if is_person_present else (0, 0, 255)
        status_text = "STATE: PERSON" if is_person_present else "STATE: EMPTY"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.imshow("ESP32-CAM Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    udp_socket.close()