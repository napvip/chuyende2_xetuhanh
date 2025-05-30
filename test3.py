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
# Thay đổi cấu hình camera để sử dụng định dạng RGB thay vì XBGR
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Cho camera khởi động

# --- THIẾT LẬP THAM SỐ ---
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng nhận diện
STOP_CLASS_ID = 0  # ID của biển báo Stop trong model (thay đổi nếu cần)
STOP_DURATION = 2  # Thời gian dừng (giây)

# --- CHƯƠNG TRÌNH CHÍNH ---
try:
    print("Xe tự hành đã khởi động...")
    forward()  # Bắt đầu di chuyển
    
    while True:
        # Chụp ảnh từ camera
        frame = picam2.capture_array()
        
        # Đảm bảo frame có đúng 3 kênh màu (RGB)
        if frame.shape[2] == 4:  # Nếu là 4 kênh (RGBA/BGRA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Phát hiện đối tượng
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Xử lý kết quả
        detected_stop = False
        current_time = time.time()
        
        # Kiểm tra các đối tượng phát hiện được
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Nếu phát hiện biển báo Stop
                if cls_id == STOP_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    detected_stop = True
                    
                    # Vẽ hộp và nhãn
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Stop {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Hiển thị khung hình (tùy chọn nếu có màn hình)
        # cv2.imshow("Camera", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        # Xử lý dừng khi phát hiện biển báo Stop
        if detected_stop and not is_stopped and (current_time - last_detection_time) > stop_cooldown:
            print("Đã phát hiện biển báo Stop! Dừng lại...")
            stop_all()
            is_stopped = True
            last_detection_time = current_time
            
            # Chờ một khoảng thời gian rồi đi tiếp
            time.sleep(STOP_DURATION)
            print("Tiếp tục di chuyển...")
            forward()
            is_stopped = False
        
        # Tránh CPU quá tải
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Dừng chương trình...")
except Exception as e:
    print(f"Lỗi: {e}")
finally:
    # Dọn dẹp
    stop_all()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    # cv2.destroyAllWindows()
    print("Đã tắt xe an toàn.")
