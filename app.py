import os
import cv2
import json
import numpy as np
import time
import threading
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from collections import Counter
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Load model
model = model_from_json(open("configs/Facial_Expression_Recognition.json", "r").read())
model.load_weights('./models/fer.h5')

face_haar_cascade = cv2.CascadeClassifier('configs/haarcascade_frontalface_default.xml')

recording = False
emotions_counter = Counter()
cap = cv2.VideoCapture(0)
predicted_emotion = "None"
faces_data = []

def detect_emotion():
    global recording, emotions_counter, predicted_emotion, faces_data
    emotions_counter.clear()
    
    while recording:
        ret, test_img = cap.read()
        if not ret:
            continue
        
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        faces_data = []
        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            if predicted_emotion != 'neutral':
                emotions_counter[predicted_emotion] += 1
            
            faces_data.append((x, y, w, h, predicted_emotion))
        
        time.sleep(1)

def update_camera():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for (x, y, w, h, emotion) in faces_data:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame = cv2.resize(frame, (200, 150))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_camera)

def start_recording(event):
    global recording
    if not recording:
        recording = True
        threading.Thread(target=detect_emotion, daemon=True).start()

def stop_recording():
    global recording
    recording = False

def save_review_to_json(product_id, review_text, emotion):
    review_data = {
        "product_id": product_id,
        "review": review_text,
        "emotion": emotion
    }
    
    try:
        if os.path.exists("reviews.json") and os.stat("reviews.json").st_size > 0:
            with open("reviews.json", "r") as file:
                try:
                    reviews = json.load(file)
                    if not isinstance(reviews, list):
                        reviews = []
                except json.JSONDecodeError:
                    reviews = []
        else:
            reviews = []
    except FileNotFoundError:
        reviews = []

    reviews.append(review_data)

    with open("reviews.json", "w") as file:
        json.dump(reviews, file, indent=4)
    
    messagebox.showinfo("Success", "Review saved successfully!")
    load_reviews()

def submit_review():
    product_id = product_id_entry.get().strip()
    user_review = review_entry.get("1.0", tk.END).strip()
    if not product_id:
        messagebox.showwarning("Warning", "Please enter a product ID.")
        return
    if not user_review:
        messagebox.showwarning("Warning", "Please enter a review.")
        return
    stop_recording()
    detected_emotion = emotions_counter.most_common(1)[0][0] if emotions_counter else "neutral"
    save_review_to_json(product_id, user_review, detected_emotion)
    emotion_label.config(text=f"Detected Emotion: {detected_emotion}")

def load_reviews():
    for widget in reviews_frame.winfo_children():
        widget.destroy()
    
    try:
        with open("reviews.json", "r") as file:
            reviews = json.load(file)
            if not isinstance(reviews, list):
                reviews = []
    except (FileNotFoundError, json.JSONDecodeError):
        reviews = []

    for review in reviews[-5:][::-1]:
        card = tk.Frame(reviews_frame, bg="#333", padx=10, pady=5, relief=tk.RIDGE, borderwidth=2)
        card.pack(fill=tk.X, pady=5)

        tk.Label(card, text=f"Product ID: {review['product_id']}", fg="white", bg="#333", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(card, text=f"Review: {review['review']}", fg="white", bg="#333", font=("Arial", 10)).pack(anchor="w")
        tk.Label(card, text=f"Emotion: {review['emotion']}", fg="white", bg="#333", font=("Arial", 10, "italic")).pack(anchor="w")

# GUI Setup
root = tk.Tk()
root.title("Product Review System")
root.geometry("600x600")
root.configure(bg='#222')

frame_main = tk.Frame(root, bg="#222")
frame_main.pack(pady=10)

frame_left = tk.Frame(frame_main, bg="#222")
frame_left.pack(side=tk.LEFT, padx=10)

frame_right = tk.Frame(frame_main, bg="#222")
frame_right.pack(side=tk.RIGHT, padx=10)

# Product ID Entry
tk.Label(frame_left, text="Enter Product ID:", fg="white", bg="#222", font=("Arial", 12, "bold")).pack(pady=5)
product_id_entry = tk.Entry(frame_left, width=50, fg="white", bg="black", insertbackground="white", font=("Arial", 12))
product_id_entry.pack(pady=5)

# Review Entry
review_entry = scrolledtext.ScrolledText(frame_left, wrap=tk.WORD, width=50, height=5, fg="white", bg="black", insertbackground="white", font=("Arial", 12))
review_entry.pack(pady=5)
review_entry.bind("<KeyPress>", start_recording)

# Camera Preview
camera_label = tk.Label(frame_right, bg="black")
camera_label.pack()
update_camera()

# Submit Button
tk.Button(frame_left, text="Submit Review", command=submit_review, fg="white", bg="#444", font=("Arial", 12, "bold")).pack(pady=10)

# Emotion Label
emotion_label = tk.Label(frame_right, text="Detected Emotion: None", font=("Arial", 12, "bold"), fg="white", bg="#222")
emotion_label.pack()

# Reviews Section
reviews_frame = tk.Frame(root, bg="#222")
reviews_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
load_reviews()

root.mainloop()
