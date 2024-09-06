import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk,ImageFont
import threading

from utils import distance_between_lms, distance_between_points
from drawing import draw_strokes, draw_point, draw_center_box
from operation import save_strokes_csv,strokes_for_saving

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
move_threshold = 0.005
save_threshold = 0.1


def is_pinching(hand_landmarks, finger_tip, pinch_threshold=0.5):
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

def is_saving_state(hand_landmarks):
    return is_pinching(hand_landmarks,RING_TIP)

def get_index_tip(hand_landmarks): 
    tip = hand_landmarks.landmark[8]
    return (tip.x, tip.y)

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
        # draw_point(frame, hand_landmarks.landmark[THUMB_TIP], color=(0, 255, 255))
        
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
            # draw_point(frame, hand_landmarks.landmark[INDEX_TIP], color=(0, 255, 0))
            if not cleared and is_clearing_state(hand_landmarks):
                strokes.clear()
                cleared = True
            # test 
            drawing = False
            # elif not drawing and is_saving_state(hand_landmarks):
            #     save_strokes_csv(strokes_for_saving, 'train1.csv')  # Save all strokes
            #     strokes.clear()  # Clear display list (optional)
        if is_saving_state(hand_landmarks):
            save_strokes_csv(strokes_for_saving, 'train1.csv')

    draw_strokes(frame, strokes)

    # Load overlay images
    overlay_img1 = cv2.imread("D:\\I4-internship\\Internship\\image\\Newindex.jpg")
    overlay_img1 = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2RGB)
    overlay_img2 = cv2.imread("D:\\I4-internship\\Internship\\image\\NewRing.jpg")
    overlay_img2 = cv2.cvtColor(overlay_img2, cv2.COLOR_BGR2RGB)
    overlay_img3 = cv2.imread("D:/I4-internship/Internship/image/Newmiddle.jpg")
    overlay_img3 = cv2.cvtColor(overlay_img3,cv2.COLOR_BGR2RGB)
    
    # Define Khmer text and font
    khmer_text = "sMeNr"
    fontpath = "fonts/limonf3.TTF"
    font_khmer = ImageFont.truetype(fontpath, 70)
    text_position = (10, 10)
    text_color = (255, 255, 255)
    center_box_frame = draw_center_box(frame, overlay_img1, overlay_img2,overlay_img3, khmer_text, font_khmer, (10, 10), text_color)

    # Resize frame for display
    h, w, _ = center_box_frame.shape
    center_box_frame_resized = cv2.resize(center_box_frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    # Convert resized frame to PIL for display
    img_pil = Image.fromarray(center_box_frame_resized)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)  
    lmain.after(10, show_frame)
    # save_strokes_csv(strokes_for_saving, 'train1.csv')

def on_closing():
    global running
    running = False
    # save_strokes_csv(strokes_for_saving, 'train1.csv')  # Save all strokes before closing
    root.destroy()
    cap.release()
    hands.close()

root = tk.Tk()
root.configure(bg='white')
lmain = tk.Label(root, bg='white')
lmain.pack()

running = True
show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

