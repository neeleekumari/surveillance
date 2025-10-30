"""Quick check if Camera 1 can be opened."""
import cv2

print("Testing Camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)

if cap.isOpened():
    print("✓ Camera 1 opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Camera 1 can read frames: {frame.shape}")
    else:
        print("✗ Camera 1 cannot read frames")
    cap.release()
else:
    print("✗ Camera 1 failed to open")

print("\nTesting Camera 0...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)

if cap.isOpened():
    print("✓ Camera 0 opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Camera 0 can read frames: {frame.shape}")
    else:
        print("✗ Camera 0 cannot read frames")
    cap.release()
else:
    print("✗ Camera 0 failed to open")
