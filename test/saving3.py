import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import threading
import os
import csv
from utils import distance_between_lms, distance_between_points
from drawing import draw_strokes, draw_point, draw_center_box
from operation import save_strokes_csv, strokes_for_saving

# Constants
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12  
RING_TIP = 16

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Video capture setup
cap = cv2.VideoCapture(0)

# Stroke variables
strokes = []
drawing = False
cleared = True
move_threshold = 0.01
save_threshold = 0.1

# Data folder
data_folder = 'D:\\I4-internship\\Internship\\data'
os.makedirs(data_folder, exist_ok=True)

# Load saved strokes
strokes_for_saving = []

# Helper functions
def is_pinching(hand_landmarks, finger_tip, pinch_threshold=0.5):
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    pinch_len = distance_between_lms(hand_landmarks.landmark[finger_tip], thumb_tip)
    index_len = distance_between_lms(hand_landmarks.landmark[INDEX_TIP], hand_landmarks.landmark[6])
    return pinch_len / index_len < pinch_threshold

def is_drawing_state(hand_landmarks):
    return is_pinching(hand_landmarks, INDEX_TIP)

def is_clearing_state(hand_landmarks):
    return is_pinching(hand_landmarks, MIDDLE_TIP)

def is_saving_state(hand_landmarks):
    return distance_between_lms(hand_landmarks.landmark[THUMB_TIP], hand_landmarks.landmark[RING_TIP]) < save_threshold

def get_index_tip(hand_landmarks): 
    tip = hand_landmarks.landmark[INDEX_TIP]
    return (tip.x, tip.y)

def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def distance_between_lms(lm1, lm2):
    return distance_between_points((lm1.x, lm1.y), (lm2.x, lm2.y))

def smooth_stroke_moving_average(points, window_size=3):
    points_array = np.array(points)
    smoothed_points = np.copy(points_array)
    for i in range(1, len(points) - 1):
        start_index = max(i - window_size // 2, 0)
        end_index = min(i + window_size // 2 + 1, len(points))
        smoothed_points[i] = np.mean(points_array[start_index:end_index], axis=0)
    return [tuple(point) for point in smoothed_points]

def save_strokes_csv(strokes, filename):
    combined_stroke = [point for stroke in strokes for point in stroke]
    formatted_data = [f'{x:.6f},{y:.6f}' for x, y in combined_stroke if isinstance((x, y), (list, tuple)) and len((x, y)) == 2]

    file_path = os.path.join(data_folder, filename)
    with open(file_path, 'w', newline='') as file:
        file.write(','.join(formatted_data) + '\n' if formatted_data else '\n')

def load_strokes_csv(filename):
    strokes = []
    file_path = os.path.join(data_folder, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                points = [(float(row[i]), float(row[i+1])) for i in range(0, len(row), 2) if i + 1 < len(row)]
                if points:
                    strokes.append(points)
    return strokes

# Load previously saved strokes
strokes_for_saving = load_strokes_csv('train2.csv')

# Main frame display function
def show_frame():
    global drawing, cleared, last_pos, current_stroke

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]

        if is_drawing_state(hand_landmarks):
            cleared = False
            tip_pos = get_index_tip(hand_landmarks)

            if not drawing:
                current_stroke = []
                strokes.append(current_stroke)
                strokes_for_saving.append(current_stroke)
                drawing = True
                last_pos = tip_pos

            if distance_between_points(tip_pos, last_pos) > move_threshold:
                current_stroke.append(tip_pos)
                last_pos = tip_pos
        else:
            if not cleared and is_clearing_state(hand_landmarks):
                strokes.clear()
                cleared = True
            elif not drawing and is_saving_state(hand_landmarks):
                save_strokes_csv(strokes_for_saving, 'train2.csv')
                strokes.clear()
            drawing = False

    draw_strokes(frame, strokes)

    # Load overlay images
    overlay_img1 = cv2.imread("D:\\I4-internship\\Internship\\image\\start1.png")
    overlay_img1 = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2RGB)
    overlay_img2 = cv2.imread("D:\\I4-internship\\Internship\\image\\save1.png")
    overlay_img2 = cv2.cvtColor(overlay_img2, cv2.COLOR_BGR2RGB)

    # Draw center box with overlays
    khmer_text = "sMeNr"
    fontpath = "fonts/limonf3.TTF"
    font_khmer = ImageFont.truetype(fontpath, 70)
    text_color = (255, 255, 255)
    center_box_frame = draw_center_box(frame, overlay_img1, overlay_img2, khmer_text, font_khmer, (10, 10), text_color)

    # Resize frame for display
    h, w, _ = center_box_frame.shape
    center_box_frame_resized = cv2.resize(center_box_frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    img_pil = Image.fromarray(center_box_frame_resized)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)  
    lmain.after(10, show_frame)
    save_strokes_csv(strokes_for_saving, 'train2.csv')

# Handle closing of the application
def on_closing():
    global running
    running = False
    save_strokes_csv(strokes_for_saving, 'train2.csv')
    root.destroy()
    cap.release()
    hands.close()

# Initialize the main application window
root = tk.Tk()
root.configure(bg='white')
lmain = tk.Label(root, bg='white')
lmain.pack()

# Start the frame display
running = True
show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
