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
    return is_pinching(hand_landmarks, RING_TIP)


def get_index_tip(hand_landmarks): 
    tip = hand_landmarks.landmark[INDEX_TIP]
    return (tip.x, tip.y)

def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def distance_between_lms(lm1, lm2):
    return distance_between_points((lm1.x, lm1.y), (lm2.x, lm2.y))
def chaikins_algorithm(points, iterations=2):
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p0 = np.array(points[i])
            p1 = np.array(points[i + 1])
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            if i == 0:
                new_points.append(tuple(p0))
            new_points.append(tuple(q))
            new_points.append(tuple(r))
            if i == len(points) - 2:
                new_points.append(tuple(p1))
        points = new_points
    return points
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

def draw_strokes(image, strokes, color=(255, 0, 0), thickness=5):
    h, w, _ = image.shape
    for stroke in strokes:
        stroke = smooth_stroke_moving_average(stroke)
        for i in range(len(stroke) - 1):
            pt1 = (int(stroke[i][0] * w), int(stroke[i][1] * h))
            pt2 = (int(stroke[i + 1][0] * w), int(stroke[i + 1][1] * h))
            cv2.line(image, pt1, pt2, color, thickness)

def draw_point(image, landmark, color=(0, 255, 0), radius=5):
    h, w, _ = image.shape
    x, y = int(landmark.x * w), int(landmark.y * h)
    cv2.circle(image, (x, y), radius, color, -1)

def draw_center_box(image, overlay_img1, overlay_img2, overlay_img3, khmer_text, font_khmer, text_position, text_color, color=(112,97,200), thickness=2):
    h, w, _ = image.shape
    padding = 20
    text_box_padding = 20
    box_width, box_height = 480, 240
    text_box_height = 180
    text_box_color = (112,97,200)
    top_left_x = padding
    top_left_y = padding
    bottom_right_x = w - padding
    bottom_right_y = h - padding

    if top_left_x + box_width > bottom_right_x:
        box_width = bottom_right_x - top_left_x
    if top_left_y + box_height > bottom_right_y:
        box_height = bottom_right_y - top_left_y

    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left_x + box_width, top_left_y + box_height)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    image_with_box = cv2.bitwise_and(image, image, mask=mask)
    cv2.rectangle(image_with_box, top_left, bottom_right, color, thickness)

    text_box_top_left = (top_left[0], bottom_right[1] + text_box_padding)
    text_box_bottom_right = (bottom_right[0], text_box_top_left[1] + text_box_height)
    if text_box_bottom_right[1] > h:
        text_box_bottom_right = (text_box_bottom_right[0], h)

    cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, -1)
    cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, thickness)

    # Resize overlay images
    overlay_size = (100, 110)  # Adjust as needed
    overlay_img1 = cv2.resize(overlay_img1, overlay_size)
    overlay_img2 = cv2.resize(overlay_img2, overlay_size)
    overlay_img3 = cv2.resize(overlay_img3, overlay_size)



    # Define the region for the first overlay
    image_top_left1 = (bottom_right[0] + padding, top_left[1])
    if image_top_left1[0] + 100 > w:
        image_top_left1 = (w - 100, image_top_left1[1])
    if image_top_left1[1] + 110 > h:
        image_top_left1 = (image_top_left1[0], h - 110)

    overlay_region1 = image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100]
    blended1 = cv2.addWeighted(overlay_img1, 1.0, overlay_region1, 0.0, 0)
    image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100] = blended1

    image_top_left2 = (image_top_left1[0], image_top_left1[1] + 110 + padding)
    if image_top_left2[0] + 100 > w:
        image_top_left2 = (w - 100, image_top_left2[1])
    if image_top_left2[1] + 110 > h:
        image_top_left2 = (image_top_left2[0], h - 110)

    overlay_region2 = image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100]
    blended2 = cv2.addWeighted(overlay_img2, 1.0, overlay_region2, 0.0, 0)
    image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100] = blended2

    image_top_left3 = (image_top_left2[0], image_top_left2[1] + 110 + padding)
    if image_top_left3[0] + 100 > w:
        image_top_left3 = (w - 100, image_top_left3[1])
    if image_top_left3[1] + 110 > h:
        image_top_left3 = (image_top_left3[0], h - 110)

    overlay_region3 = image_with_box[image_top_left3[1]:image_top_left3[1] + 110, image_top_left3[0]:image_top_left3[0] + 100]
    blended3 = cv2.addWeighted(overlay_img3, 1.0, overlay_region3, 0.0, 0)
    image_with_box[image_top_left3[1]:image_top_left3[1] + 110, image_top_left3[0]:image_top_left3[0] + 100] = blended3

    img_pil = Image.fromarray(image_with_box)
    draw = ImageDraw.Draw(img_pil)
    left_text_position = (30, (h // 2) + 60)
    draw.text(left_text_position, khmer_text, font=font_khmer, fill=text_color)

    return np.array(img_pil)







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
            draw_point(frame, hand_landmarks.landmark[INDEX_TIP], color=(0, 255, 0))
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
            drawing = False    
               # Save strokes only when the ring finger and thumb are pinching
        if is_saving_state(hand_landmarks):
            save_strokes_csv(strokes_for_saving, 'train2.csv')

    draw_strokes(frame, strokes)

    # Load overlay images
    overlay_img1 = cv2.imread("D:\\I4-internship\\Internship\\image\\start1.png")
    overlay_img1 = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2RGB)
    overlay_img2 = cv2.imread("D:\\I4-internship\\Internship\\image\\save1.png")
    overlay_img2 = cv2.cvtColor(overlay_img2, cv2.COLOR_BGR2RGB)
    overlay_img3 = cv2.imread("D:/I4-internship/Internship/image/Newmiddle.jpg")
    overlay_img3 = cv2.cvtColor(overlay_img3,cv2.COLOR_BGR2RGB)

    # Draw center box with overlays
    khmer_text = "sMeNr"
    fontpath = "fonts/limonf3.TTF"
    font_khmer = ImageFont.truetype(fontpath, 70)
    text_color = (255, 255, 255)
    center_box_frame = draw_center_box(frame, overlay_img1, overlay_img2,overlay_img3, khmer_text, font_khmer, (10, 10), text_color)

    # Resize frame for display
    h, w, _ = center_box_frame.shape
    center_box_frame_resized = cv2.resize(center_box_frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    img_pil = Image.fromarray(center_box_frame_resized)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)  
    lmain.after(10, show_frame)
    # save_strokes_csv(strokes_for_saving, 'train2.csv')


# Handle closing of the application
def on_closing():
    global running
    running = False
    # save_strokes_csv(strokes_for_saving, 'train2.csv')
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
