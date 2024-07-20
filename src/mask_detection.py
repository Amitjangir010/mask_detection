import cv2
import numpy as np
import tensorflow as tf

def detect_and_predict_mask(face, model):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = tf.keras.preprocessing.image.img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    mask_prediction = model.predict(face)
    return mask_prediction[0][0]

def get_face_id(face_encoding):
    global current_id
    if len(known_face_encodings) == 0:
        known_face_encodings.append(face_encoding)
        known_face_ids.append(current_id)
        current_id += 1
        return current_id - 1