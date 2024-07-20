import cv2
import numpy as np
import face_recognition
import threading
from tensorflow.keras.models import load_model
from mask_detection import detect_and_predict_mask
from social_distancing import calculate_intersection_area
from utils import setup_logger

class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.ret = False
        self.frame = None
        self.stopped = False

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True
        self.cap.release()

def get_face_id(face_encoding, known_face_encodings, known_face_ids):
    global current_id
    if len(known_face_encodings) == 0:
        known_face_encodings.append(face_encoding)
        known_face_ids.append(current_id)
        current_id += 1
        return current_id - 1

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        return known_face_ids[best_match_index]
    else:
        known_face_encodings.append(face_encoding)
        known_face_ids.append(current_id)
        current_id += 1
        return current_id - 1

if __name__ == "__main__":
    mask_model = load_model('models/mask_detector_model.h5')
    logger = setup_logger()
    logger.info("Mask detection model loaded")

    video_thread = VideoCaptureThread()
    video_thread.start()
    logger.info("Video capture thread started")

    known_face_encodings = []
    known_face_ids = []
    current_id = 0
    log_data = []
    status_tracker = {}

    INTERSECTION_THRESHOLD = 50

    while True:
        if not video_thread.ret:
            continue
        frame = cv2.resize(video_thread.frame, (800, 600))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face = frame[top:bottom, left:right]
            face_id = get_face_id(face_encoding, known_face_encodings, known_face_ids)
            mask_prediction = detect_and_predict_mask(face, mask_model)

            mask_status = 'Mask' if mask_prediction < 0.5 else 'No-Mask'

            if face_id not in status_tracker or status_tracker[face_id] != mask_status:
                status_tracker[face_id] = mask_status
                log_data.append((face_id, mask_status))
                logger.info(f"ID: {face_id} - {mask_status}")

            label = f"ID: {face_id} - {mask_status}"
            color = (0, 255, 0) if mask_status == 'Mask' else (0, 0, 255)

            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        for i in range(len(face_locations)):
            for j in range(i + 1, len(face_locations)):
                box1 = (face_locations[i][3], face_locations[i][0], face_locations[i][1], face_locations[i][2])
                box2 = (face_locations[j][3], face_locations[j][0], face_locations[j][1], face_locations[j][2])
                intersection_area = calculate_intersection_area(box1, box2)
                if intersection_area > INTERSECTION_THRESHOLD:
                    cv2.putText(frame, "Please maintain social distancing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    logger.info("Social distancing warning displayed")

        y0, dy = 50, 20
        for i, (face_id, status) in enumerate(log_data[-15:]):
            cv2.putText(frame, f"ID: {face_id} - {status}", (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video_thread.stop()
    cv2.destroyAllWindows()
    logger.info("Video capture thread stopped and application closed")
