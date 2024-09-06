import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os
import csv 

cap = cv2.VideoCapture(0) 

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils 

strokes = []
drawing = False
cleared = True
move_threshold = 0.01

def is_pinching(hand_landmarks, finger_tip, pinch_threshold=0.7):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    index_pip = hand_landmarks.landmark[6]
    index_len = distance_between_lms(index_tip, index_pip)
    pinch_len = distance_between_lms(hand_landmarks.landmark[finger_tip], thumb_tip)
    return pinch_len / index_len < pinch_threshold

def is_drawing_state(hand_landmarks):
    return is_pinching(hand_landmarks, INDEX_TIP)

def is_clearing_state(hand_landmarks):
    return is_pinching(hand_landmarks, MIDDLE_TIP)

def get_index_tip(hand_landmarks):
    tip = hand_landmarks.landmark[8]
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
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    points_array = np.array(points)
    smoothed_points = np.copy(points_array)
    for i in range(1, len(points) - 1):
        start_index = max(i - window_size // 2, 0)
        end_index = min(i + window_size // 2 + 1, len(points))
        smoothed_points[i] = np.mean(points_array[start_index:end_index], axis=0)
    return [tuple(point) for point in smoothed_points]

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

# Define Khmer text
khmer_text = "sMeNr"
fontpath = "fonts/limonf3.TTF"  # Update this to the correct path of your Khmer font
font_khmer = ImageFont.truetype(fontpath, 70)  # Adjust the size as needed 
text_position = (10, 10)  # Adjust as needed
text_color = (255, 255, 255)

def draw_center_box(image, overlay_img1, overlay_img2, khmer_text, font_khmer, text_position, text_color, color=(25, 167, 206), thickness=2):
    h, w, _ = image.shape
    padding = 20

    text_box_padding = 40  # Additional padding for the text box
    box_width, box_height = 480, 240
    text_box_height = 150
    text_box_color = (25, 167, 206)

    # Calculate position for the center box
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

    # Create a mask for the center box
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    # Apply mask to image
    image_with_box = cv2.bitwise_and(image, image, mask=mask)
    
    # Draw the center box
    
    cv2.rectangle(image_with_box, top_left, bottom_right, color, thickness)

    # Draw the additional text box below the center box with more padding
    text_box_top_left = (top_left[0], bottom_right[1] + text_box_padding)
    text_box_bottom_right = (bottom_right[0], text_box_top_left[1] + text_box_height)

    if text_box_bottom_right[1] > h:  # Ensure the text box doesn't go out of bounds
        text_box_bottom_right = (text_box_bottom_right[0], h)

    # Fill the text box with white color
    cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, -1)
    cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, thickness)

    # Load and place overlay images
    image_top_left1 = (bottom_right[0] + padding, top_left[1])
    if image_top_left1[0] + 100 > w:
        image_top_left1 = (w - 100, image_top_left1[1])
    if image_top_left1[1] + 110 > h:
        image_top_left1 = (image_top_left1[0], h - 110)

    overlay_region1 = image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100]
    blended1 = cv2.addWeighted(overlay_img1, 1.0, overlay_region1, 0.0, 0)
    image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100] = blended1

    image_top_left2 = (image_top_left1[0], image_top_left1[1] + 110 + 20)
    if image_top_left2[0] + 100 > w:
        image_top_left2 = (w - 100, image_top_left2[1])
    if image_top_left2[1] + 110 > h:
        image_top_left2 = (image_top_left2[0], h - 110)

    overlay_region2 = image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100]
    blended2 = cv2.addWeighted(overlay_img2, 1.0, overlay_region2, 0.0, 0)
    image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100] = blended2

    # Convert to PIL for text drawing
    image_pil = Image.fromarray(image_with_box)
    draw = ImageDraw.Draw(image_pil)
    draw.text(text_position, khmer_text, font=font_khmer, fill=text_color)

    # Convert back to OpenCV format
    image_with_text = np.array(image_pil)
    return image_with_text

# Load overlay images once outside the loop
overlay_img1 = cv2.imread("D:\\I4-internship\\Internship\\image\\start1.png")
overlay_img1 = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2RGB)
overlay_img1 = cv2.resize(overlay_img1, (100, 110))

overlay_img2 = cv2.imread("D:\\I4-internship\\Internship\\image\\save1.png")
overlay_img2 = cv2.cvtColor(overlay_img2, cv2.COLOR_BGR2RGB)
overlay_img2 = cv2.resize(overlay_img2, (100, 110))

def show_frame():
    global drawing
    global last_pos
    global current_stroke
    global cleared

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]
        draw_point(frame, hand_landmarks.landmark[THUMB_TIP], color=(0, 255, 255))
        
        if is_drawing_state(hand_landmarks):
            cleared = False
            draw_point(frame, hand_landmarks.landmark[INDEX_TIP], color=(255, 0, 0))
            tip_pos = get_index_tip(hand_landmarks)
            
            if not drawing:
                current_stroke = []
                strokes.append(current_stroke)
                drawing = True
                last_pos = tip_pos

            if distance_between_points(tip_pos, last_pos) > move_threshold:
                current_stroke.append(tip_pos)
                last_pos = tip_pos
        else:
            draw_point(frame, hand_landmarks.landmark[INDEX_TIP], color=(0, 255, 0))
            if not cleared and is_clearing_state(hand_landmarks):
                strokes.clear()
                cleared = True
            drawing = False

    # Draw strokes (optional: consider drawing only when needed)
    draw_strokes(frame, strokes)

    # Draw center box and overlay images
    center_box_frame = draw_center_box(frame, overlay_img1, overlay_img2, khmer_text, font_khmer, text_position, text_color, color=(25, 167, 206))

    # Resize the entire processed frame to 200% (if necessary)
    h, w, _ = center_box_frame.shape
    new_dim = (w * 2, h * 2)
    center_box_frame_resized = cv2.resize(center_box_frame, new_dim, interpolation=cv2.INTER_LINEAR)

    # Convert resized frame to PIL for display
    img_pil = Image.fromarray(center_box_frame_resized)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)  
    lmain.after(10, show_frame)

def save_strokes():
    if strokes:  # Only save if there are strokes to save
        save_strokes_csv(strokes, 'train1.csv')

def save_strokes_csv(strokes, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for stroke in strokes:
            writer.writerow(['x', 'y'])
            for point in stroke:
                writer.writerow(point)

# GUI Setup
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")

# Create a Label to display the video feed
lmain = tk.Label(root)
lmain.pack()

# Add a Save Button
save_button = tk.Button(root, text="Save Strokes", command=save_strokes)
save_button.pack()

# Start the video feed
show_frame()

root.mainloop()
