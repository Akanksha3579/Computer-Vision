import cv2
import pyautogui
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import streamlit as st
from math import hypot

#pip install screen-brightness-control


# Initialize Streamlit interface
st.title("Hand Gesture Control for Volume and Brightness")
st.text("Use hand gestures to control the volume and brightness.")
run = st.checkbox("Run")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Streamlit Frame
frame_window = st.image([])

try:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image
        results = hands.process(image_rgb)

        # List to store landmarks
        lmList = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract hand landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Volume Control: Based on hand gesture (index finger above or below thumb)
                if lmList:
                    index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                    if index_finger_y < thumb_y:
                        pyautogui.press('volumeup')
                        st.write("Volume Up")
                    elif index_finger_y > thumb_y:
                        pyautogui.press('volumedown')
                        st.write("Volume Down")

                    # Brightness Control: Based on distance between thumb and index finger
                    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                    x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
                    cv2.circle(frame, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    length = hypot(x2 - x1, y2 - y1)
                    brightness = np.interp(length, [15, 220], [0, 100])
                    sbc.set_brightness(int(brightness))
                    st.write(f"Brightness: {int(brightness)}, Length: {length}")

        # Display the resulting frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

except KeyboardInterrupt:
    st.write("Interrupted by user")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
