import time
import random
import numpy as np
from ultralytics import YOLO
import cv2
import threading
import os
import sys

# Check if tkinter is available
try:
    import tkinter as tk
    from tkinter import ttk, font
    from PIL import Image, ImageTk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception as e:
    print(f"Error initializing tkinter: {e}")
    TKINTER_AVAILABLE = False


class TrafficSignRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Load model
        self.model = YOLO('best.pt')  # Update path to your model
        self.class_names = self.model.names
        self.colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                       for i in self.class_names}

        # Setup camera
        self.cap = None  # Will be initialized when starting
        self.camera_id = 0  # Default camera ID (usually webcam)

        # GUI variables
        self.is_running = False
        self.detection_thread = None
        self.current_frame = None
        self.detection_results = []

        self.create_gui()

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Detection",
                                       command=self.toggle_detection)
        self.start_button.pack(pady=10, padx=10, fill=tk.X)

        # Confidence threshold
        ttk.Label(control_frame, text="Confidence Threshold:").pack(pady=(10, 0), padx=10)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=1.0,
                               variable=self.conf_threshold, orient=tk.HORIZONTAL)
        conf_scale.pack(pady=5, padx=10, fill=tk.X)

        # Fix the textvariable issue
        self.conf_value = tk.StringVar(value="0.50")
        conf_label = ttk.Label(control_frame, textvariable=self.conf_value)
        conf_label.pack(pady=0, padx=10)

        # Update label when scale changes
        def update_conf_label(*args):
            self.conf_value.set(f"{self.conf_threshold.get():.2f}")

        self.conf_threshold.trace_add("write", update_conf_label)

        # Results panel
        results_frame = ttk.LabelFrame(control_frame, text="Detection Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        self.results_text = tk.Text(results_frame, height=10, width=25, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

    def toggle_detection(self):
        if self.is_running:
            self.stop_detection()
            self.start_button.config(text="Start Detection")
        else:
            self.start_detection()
            self.start_button.config(text="Stop Detection")

    def start_detection(self):
        self.is_running = True
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.is_running = False
            return

        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def stop_detection(self):
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def detection_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            # Run detection
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold.get(),
                iou=0.45,
                stream=True,
                verbose=False
            )

            # Process results
            self.detection_results = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_name = self.class_names[cls_id]

                    self.detection_results.append({
                        'label': label_name,
                        'confidence': conf,
                        'box': (x1, y1, x2, y2),
                        'color': self.colors.get(cls_id, (0, 255, 0))
                    })

                    # Draw on frame
                    color = self.colors.get(cls_id, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label_name} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

            # Convert to RGB for tkinter
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_gui()

    def update_gui(self):
        if self.current_frame is not None and self.is_running:
            # Update canvas with new frame
            img = Image.fromarray(self.current_frame)
            img = ImageTk.PhotoImage(image=img)
            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img  # Keep reference

            # Update results text
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            if self.detection_results:
                for result in self.detection_results:
                    self.results_text.insert(
                        tk.END,
                        f"{result['label']}: {result['confidence']:.2f}\n"
                    )
            else:
                self.results_text.insert(tk.END, "No signs detected")

            self.results_text.config(state=tk.DISABLED)

        # Schedule next update
        if self.is_running:
            self.root.after(30, self.update_gui)

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()


class ConsoleTrafficSignRecognition:
    """Fallback console-based traffic sign recognition when GUI is unavailable"""
    def __init__(self):
        # Load model
        self.model = YOLO('best.pt')  # Update path to your model
        self.class_names = self.model.names
        self.is_running = False
        self.conf_threshold = 0.5
        self.camera_id = 0
        self.cap = None

    def start(self):
        print("Starting Traffic Sign Recognition in console mode...")
        print("Press 'q' to quit at any time")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
                
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=0.45,
                verbose=False
            )
            
            # Process and display results
            detected_signs = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_name = self.class_names[cls_id]
                    detected_signs.append((label_name, conf))
                    
                    # Draw on frame for visualization
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label_name} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)
            
            # Display results in console and show frame
            if detected_signs:
                print("\nDetected signs:")
                for label, conf in detected_signs:
                    print(f"  - {label}: {conf:.2f}")
            else:
                print("\rNo signs detected", end="")
                
            # Show the frame
            cv2.imshow("Traffic Sign Recognition", frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
        
        # Clean up
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def fix_tkinter_instructions():
    """Print instructions for fixing tkinter"""
    print("=" * 80)
    print("Tkinter initialization error detected!")
    print("=" * 80)
    print("\nTo fix this issue, you need to install Tcl/Tk properly.")
    print("\nOption 1: Install ActiveTcl")
    print("  1. Download ActiveTcl from https://www.activestate.com/products/tcl/downloads/")
    print("  2. Install it and restart your application")
    
    print("\nOption 2: Reinstall Python with tcl/tk")
    print("  - When installing Python, make sure to check 'tcl/tk and IDLE'")
    
    print("\nOption 3: For Anaconda users")
    print("  - Run: conda install -c anaconda tk")
    
    print("\nRunning in console mode as fallback...\n")


if __name__ == "__main__":
    if TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            app = TrafficSignRecognitionGUI(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            fix_tkinter_instructions()
            console_app = ConsoleTrafficSignRecognition()
            console_app.start()
    else:
        fix_tkinter_instructions()
        console_app = ConsoleTrafficSignRecognition()
        console_app.start()
