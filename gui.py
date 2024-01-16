import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

def DrowsinessDetectionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer="adam", loss=["binary_crossentropy"], metrics=["accuracy"])
    return model

def eye_state_prediction(eye_roi):
    eye_roi = cv2.resize(eye_roi, (48, 48))
    eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eye_roi = np.expand_dims(eye_roi, axis=-1)
    model = DrowsinessDetectionModel("model_a.json", "model_weights.h5")
    prediction = model.predict(np.expand_dims(eye_roi, axis=0))  # Add an extra dimension for batch
    print("Prediciton value: ",prediction)
    if prediction > 0.5:
        return "Open"
    else:
        return "Closed"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eyec.detectMultiScale(roi_frame)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_frame[ey:ey+eh, ex:ex+ew]
            eye_state = eye_state_prediction(eye_roi)
            color = (0, 255, 0) if eye_state == "Open" else (0, 0, 255)
            cv2.rectangle(roi_frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(roi_frame, eye_state, (ex, ey - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()