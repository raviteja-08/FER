import os
import cv2
import json
import numpy as np
import time
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from collections import Counter
import tkinter as tk
from tkinter import scrolledtext, messagebox

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
            if predicted_emotion!='neutral':
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
    
    messagebox.showinfo("Success", "Review saved successfully!")

def submit_review():
    user_review = review_entry.get("1.0", tk.END).strip()
    if not user_review:
        messagebox.showwarning("Warning", "Please enter a review.")
        return
    detected_emotion = detect_emotion()
    save_review_to_json(user_review, detected_emotion)
    emotion_label.config(text=f"Detected Emotion: {detected_emotion}")

# GUI Setup
root = tk.Tk()
root.title("Product Review System")
root.geometry("500x400")
root.configure(bg='#222')  # Dark background

# Label with white text
tk.Label(root, text="Enter your product review:", fg="white", bg="#222", font=("Arial", 12, "bold")).pack(pady=5)

# Text box with white text and black background
review_entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5, fg="white", bg="black", insertbackground="white", font=("Arial", 12))
review_entry.pack(pady=5)

# Submit button
submit_button = tk.Button(root, text="Submit Review", command=submit_review, fg="white", bg="#444", font=("Arial", 12, "bold"))
submit_button.pack(pady=10)

# Emotion label
emotion_label = tk.Label(root, text="Detected Emotion: None", font=("Arial", 12, "bold"), fg="white", bg="#222")
emotion_label.pack(pady=5)

root.mainloop()
