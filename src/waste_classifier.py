import cv2
from ultralytics import YOLO
import serial
import time

# =============================
# 1. Load YOLO trained model
# =============================
model = YOLO(r"d:/vscode/best.pt")  # path to your trained weights

# =============================
# 2. Open laptop webcam
# =============================
cap = cv2.VideoCapture(0)  # 0 = default laptop webcam
if not cap.isOpened():
    print("❌ Error: Could not access webcam")
    exit()

# =============================
# 3. Connect to ESP32 via Serial
# =============================
try:
    ser = serial.Serial("COM3", 115200, timeout=1)  # change COM port if needed
    time.sleep(2)  # wait for ESP32 reset
    print("✅ Connected to ESP32")
except Exception as e:
    print(f"❌ Error: Could not connect to ESP32 → {e}")
    ser = None

# =============================
# 4. Real-time Loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO classification
    results = model.predict(frame, imgsz=128, verbose=False)

    if results and results[0].probs is not None:
        # Get top prediction
        class_id = results[0].probs.top1
        pred_class = results[0].names[class_id]
        confidence = results[0].probs.top1conf.item()

        # Display result on frame
        cv2.putText(frame, f"{pred_class} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send result to ESP32
        if ser:
            msg = pred_class.strip() + "\n"
            ser.write(msg.encode())
            print(f"📤 Sent to ESP32: {pred_class} ({confidence:.2f})")

    # Show webcam output
    cv2.imshow("Waste Classification", frame)

    # Exit on "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# 5. Cleanup
# =============================
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
