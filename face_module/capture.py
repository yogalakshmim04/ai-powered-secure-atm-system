# face_module/capture.py
import cv2
import os

def capture_faces(account_number, save_path="uploads"):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    user_folder = os.path.join(save_path, str(account_number))
    os.makedirs(user_folder, exist_ok=True)

    count = 0
    print("[INFO] Look at the camera. Capturing 3 photos...")

    while count < 3:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            filename = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Captured: {count}/3", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Face Capture - Press Q to quit", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return count == 3   # True if all 3 captured