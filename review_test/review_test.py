import os
import cv2
import json
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from collections import Counter
import time

# Load model
model = model_from_json(open("../configs/Facial Expression Recognition.json", "r").read())
model.load_weights('../models/fer.h5')

face_haar_cascade = cv2.CascadeClassifier('../configs/haarcascade_frontalface_default.xml')

def detect_emotion():
    cap = cv2.VideoCapture(0)
    emotions_counter = Counter()
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Capture emotions for 10 seconds
        ret, test_img = cap.read()
        if not ret:
            continue
        
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y+w, x:x+h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            emotions_counter[predicted_emotion] += 1
        
        time.sleep(1)  # Capture emotion every second
    
    cap.release()
    cv2.destroyAllWindows()
    
    return emotions_counter.most_common(1)[0][0] if emotions_counter else "neutral"


def save_review_to_json(review_text, emotion):
    review_data = {
        "review": review_text,
        "emotion": emotion
    }
    
    with open("../reviews.json", "a") as file:
        json.dump(review_data, file, indent=4)
        file.write("\n")
    
    print("Review saved successfully!")

if __name__ == "__main__":
    user_review = input("Enter your product review: ")
    detected_emotion = detect_emotion()
    save_review_to_json(user_review, detected_emotion)
    print(f"Detected Emotion: {detected_emotion}")
