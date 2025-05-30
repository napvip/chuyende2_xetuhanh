import time
import random
import RPi.GPIO as GPIO
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# --- CẤU HÌNH GPIO ---
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Chân điều khiển motor (BOARD numbering)
IN1 = 11  # Trái
IN2 = 13
IN3 = 15  # Phải
IN4 = 12
ENA = 16  # PWM trái
ENB = 18  # PWM phải

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(100)
pwmB.start(100)

# --- ĐIỀU KHIỂN ---
def stop_all():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)   # Trái lùi
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)    # Phải tiến

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)    # Trái tiến
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)   # Phải lùi

# --- TRẠNG THÁI VÀ TIMER ---
is_stopped = False
last_detection_time = 0
stop_cooldown = 3  # giây để tránh detect liên tục

# --- TẢI MODEL YOLO ---
model = YOLO('/home/toan/xetuhanh/chuyende2_xetuhanh/best.pt')
class_names = model.names

# --- KHỞI TẠO CAMERA ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

# --- HÀM XỬ LÝ KHI PHÁT HIỆN STOP ---
def handle_stop_detection():
    global is_stopped, last_detection_time
    current_time = time.time()
    
    # Kiểm tra cooldown để tránh xử lý liên tục
    if current_time - last_detection_time < stop_cooldown:
        return
    
    print("🛑 PHÁT HIỆN BIỂN BÁO STOP - DỪNG XE!")
    stop_all()
    is_stopped = True
    last_detection_time = current_time

# --- VÒNG LẶP CHÍNH ---
try:
    print("🚗 Bắt đầu chạy xe và nhận diện biển báo...")
    
    while True:
        # Capture frame từ camera
        frame = picam2.capture_array()
        
        # Chuyển đổi từ 4 kênh sang 3 kênh nếu cần
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        try:
            # Chạy YOLO detection
            results = model.predict(source=frame, conf=0.6, iou=0.45, stream=True, verbose=False)
            
            stop_detected = False
            
            # Xử lý kết quả detection
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Lấy thông tin bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = class_names[cls_id].lower()
                        
                        # Kiểm tra nếu phát hiện biển "stop"
                        if "stop" in label:
                            stop_detected = True
                            
                            # Vẽ bounding box màu đỏ cho biển stop
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"STOP {confidence:.2f}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Vẽ bounding box màu xanh cho các object khác
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Xử lý khi phát hiện biển stop
            if stop_detected:
                handle_stop_detection()
            elif not is_stopped:
                # Nếu không phát hiện stop và xe chưa dừng thì tiếp tục chạy
                forward()
        
        except Exception as e:
            print(f"⚠️ Lỗi khi chạy detection: {e}")
            # Tiếp tục chạy xe nếu có lỗi detection
            if not is_stopped:
                forward()
        
        # Hiển thị trạng thái trên frame
        status_text = "STOPPED" if is_stopped else "RUNNING"
        status_color = (0, 0, 255) if is_stopped else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Hiển thị frame
        cv2.imshow("Traffic Sign Detection", frame)
        
        # Nhấn 'q' để thoát, 'r' để reset và chạy lại
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("🔄 Reset - Xe bắt đầu chạy lại...")
            is_stopped = False
            forward()

except KeyboardInterrupt:
    print("\n⛔ Dừng chương trình bằng Ctrl+C")

finally:
    # Dọn dẹp tài nguyên
    print("🧹 Dọn dẹp tài nguyên...")
    picam2.stop()
    cv2.destroyAllWindows()
    stop_all()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("✅ Hoàn tất!")