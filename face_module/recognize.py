# face_module/recognize.py
import cv2

def recognize_face(expected_label, model_path="models/lbph_model.yml", threshold=80):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    result = False
    attempts = 0

    while attempts < 30:  # try for ~30 frames
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)

            print(f"[DEBUG] label={label}, confidence={confidence:.1f}")

            if label == expected_label and confidence < threshold:
                result = True
                break

        if result:
            break
        attempts += 1
        cv2.imshow("Face Login - Please look at camera", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return result