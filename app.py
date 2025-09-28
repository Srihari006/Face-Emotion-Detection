import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- Paths & Model ----------------
base_dir = "dataset/images"
train_dir = os.path.join(base_dir, "train")
model_path = "emotion_model.keras"
img_size = (48, 48)

@st.cache_resource
def load_emotion_model():
    return load_model(model_path)

model = load_emotion_model()
class_labels = sorted(os.listdir(train_dir))

# ---------------- Prediction Helper ----------------
def predict_image(img):
    """Predict emotion from a BGR or RGB image array"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, img_size)
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=(0,-1))  # Add batch & channel dimensions
    preds = model.predict(gray)
    return preds

# ---------------- Streamlit UI ----------------
st.title("😊 Face Emotion Recognition")
menu = ["Upload Image", "Webcam"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- Image Upload ----------------
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        try:
            # Open image safely and convert to RGB
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)  # Convert to numpy array
            
            preds = predict_image(img_array)
            pred_label = class_labels[np.argmax(preds)]
            
            st.image(image, caption=f"Prediction: {pred_label}", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

# ---------------- Webcam ----------------
elif choice == "Webcam":
    st.write("### 🎥 Real-time Webcam Detection")

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    class EmotionDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, img_size)
                roi_gray = roi_gray.astype("float32") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=(0,-1))
                preds = model.predict(roi_gray)[0]
                emotion = class_labels[np.argmax(preds)]

                # Draw rectangle and label
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
            return img

    webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)
