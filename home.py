import streamlit as st
import cv2
import numpy as np
from numpy import asarray
from typing import Union, Tuple, Dict
import pandas as pd

from PIL import Image
import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from utils import (
    CNN,
    start_webcam_feed,
    load_model,
    create_bokeh_plot,
    dataframe_to_dict,
    label_mapping,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GLOBAL_PATH = os.getcwd()
MODEL_PATH = GLOBAL_PATH + "\model\emotion_detector.pt"
net = load_model(MODEL_PATH, device)


# if __name__ == 'main':
background_image_path = 'https://images.unsplash.com/photo-1579546929518-9e396f3cc809?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
projects_background_custom_css = f"""
<style>
    /* Set background color for main content area */
    .st-emotion-cache-1r4qj8v {{
        background-image: url("{background_image_path}");
    }}
    .st-emotion-cache-18ni7ap {{
        background: rgba(255, 255, 255, 0.1);
    }}
    .st-emotion-cache-10trblm {{
        font-size: 60px;
        color: rgba(232, 237, 250, 0.9);
        font-weight: bold;
        }}
    .st-emotion-cache-40ynm6.eczjsme8 {{
        background-color: rgb(48,60,77, 0.1);
    }}
    .st-emotion-cache-m8hsoe eczjsme9 {{
        background-color: rgb(48,60,77, 0.1);
    }}
    .st-emotion-cache-vk3wp9 {{
    background-color: rgba(232, 237, 250, 0.55) !important;
    overflow: hidden !important;
    }}
</style>
"""

emoji_animation_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    /* Set background color for main content area */
    .st-emotion-cache-1r4qj8v {{
        background-image: url("{background_image_path}");
    }}
    .st-emotion-cache-18ni7ap {{
        background: rgba(255, 255, 255, 0.01);
    }}
    /* Div containing the page title */
    .st-emotion-cache-10trblm {{
        font-size: 60px;
        color: rgba(232, 237, 250, 0.9);
        font-weight: bold;
    }}
    .st-emotion-cache-40ynm6.eczjsme8 {{
        background-color: rgb(48,60,77, 0.1);
    }}
    /* Borders of the sidebar */
    .st-emotion-cache-vk3wp9 {{
    box-shadow: rgba(232, 237, 250, 0.55)  -2rem 0px 2rem 2rem;
    }}
    .st-emotion-cache-m8hsoe eczjsme9 {{
        background-color: rgb(48,60,77, 0.1);
    }}
    /* Emoji Animation */
    .st-emotion-cache-vk3wp9 {{
        background-color: rgba(232, 237, 250, 0.55) !important;
        overflow: hidden !important;
    }}
    /* Emoji Animation */
    #animation-container {{
        font-size: 50px;
        text-align: center;
    }}

    .emoji {{
        display: inline-block;
        animation: wave 2.5s ease-out;
    }}

    @keyframes wave {{
        0% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-30px); }}
        100% {{ transform: translateY(0); }}
    }}

    .emoji:nth-child(1) {{
        animation-delay: 0s;
    }}

    .emoji:nth-child(2) {{
        animation-delay: 0.1s;
    }}

    .emoji:nth-child(3) {{
        animation-delay: 0.2s;
    }}

    .emoji:nth-child(4) {{
        animation-delay: 0.3s;
    }}
    .st-emotion-cache-1lypi3u {{
        text-align: center;
    }}
</style>
<script>
</script>
</head>
<body>
<div id="animation-container">
    <span class="app_title" style = "font-size: 60px; color: rgba(232, 237, 250, 0.9);
        font-weight: bold;"> MoodMapper </span>
    <span class="emoji">😊</span>
    <span class="emoji">😄</span>
    <span class="emoji">😃</span>
    <span class="emoji">😁</span>
</div>
</body>
</html>
"""


with st.sidebar:
    st.header("Welcome to MoodMapper")
    st.divider()
    st.subheader("The app that helps you keep smiling !")
    st.write("""MoodMapper detects your emotions in real-time through your camera feed.""")
    st.write("""It then generates an emotion report to analyze and enhance your mood.""")
    st.write("""Press the start button when you're ready, and click stop whenever you fancy.""")


page_style = st.markdown(emoji_animation_html, unsafe_allow_html=True)

label_mapping=label_mapping
st.divider()

col1, col2 = st.columns(2)

control_panel = st.empty()

with control_panel:
    with col1:
        start_button = st.button(
            "Start webcam",
            key='start_webcam_button',
            on_click=start_webcam_feed,
        )
    
    with col2:
        stop_button = st.button(
            "Stop webcam",
            key='stop_webcam_button',
        )

st.write("")
st.write("")
camera_feed_placeholder = st.empty()
assessment_placeholder = st.empty()

face_cascade_name = GLOBAL_PATH + '/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(face_cascade_name)
cap = cv2.VideoCapture(0)

if "run" not in st.session_state:
    st.session_state['run'] = False


session_counts = {label: 0 for label in label_mapping.keys()}  # Initialize counts for each label

def count_sessions(cap: cv2.VideoCapture, net: torch.nn.Module, label_mapping: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Counts sessions and detects emotions from faces captured by a video feed.

    Args:
    - cap (cv2.VideoCapture): Video capture object for accessing frames.
    - net (torch.nn.Module): Pre-trained neural network for emotion detection.
    - label_mapping (Dict[str, int]): Mapping of emotion labels to indices.

    Returns:
    - frame (np.ndarray): Processed frame with emotion detection annotations.
    - session_counts (Dict[str, int]): Dictionary containing counts of detected emotions in the session.
    """
    session_counts = {label: 0 for label in label_mapping.keys()}  # Initialize counts for each label
    
    _, frame = cap.read()
    frame = frame[50:700, 50:700, :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detection.detectMultiScale(gray, 1.1, 4)

    # Process each detected face
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x + 10, y + 10), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        face_gray = gray[y:y + h, x:x + w]
        img = cv2.resize(face_gray, (48, 48))
        img = np.expand_dims(np.expand_dims(np.float32(img), axis=0), axis=0)
        final_image = torch.from_numpy(img)
        output = net(final_image)
        pred_proba = F.softmax(output, dim=1)
        emotion = torch.argmax(pred_proba, 1)
        label_idx = emotion.item()
        label = list(label_mapping.keys())[list(label_mapping.values()).index(label_idx)]
        prob = pred_proba.amax()
        x = str(round(prob.item() * 100, 2)) + " %" # Convert probability to percentage
        DisplayText = "{}: {}".format(label, x)

        cv2.putText(frame, DisplayText, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        session_counts[label] += 1
        
    return frame, session_counts


session_counts_list = {}

while st.session_state['run'] and cap.isOpened():
    if stop_button:
        df = dataframe_to_dict(st.session_state.output)
        main_emotion_string = df.iloc[df['Percentage'].idxmax()].get('Emotion')
        main_emotion_value = df.iloc[df['Percentage'].idxmax()].get('Percentage_%')
        bokeh_plot = create_bokeh_plot(df)

        camera_feed_placeholder.bokeh_chart(
            bokeh_plot,
            use_container_width=True
        )

        assessment_placeholder.markdown(
            f"<span style='font-size: larger; font-weight: bold;'>During this session, you were {main_emotion_string} most of the time ! ({main_emotion_value})</span>",
            unsafe_allow_html=True
        )   
        cap.release()
        cv2.destroyAllWindows() 
        break
    else:
        try:
            frame, session_counts = count_sessions(cap, net, label_mapping)
            camera_feed_placeholder.image(frame, channels="BGR")
            session_counts_list = {k: session_counts_list.get(k, 0) + session_counts.get(k, 0) for k in set(session_counts_list) | set(session_counts)}
            st.session_state.output = session_counts_list
            
        except Exception as e:
            continue


