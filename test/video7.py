import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

strokes = []
drawing = False
cleared = True
#move_threshold = 0.02  # Increased for smoother tracking
move_threshold = 0.02

# Finger pinching detection with a refined threshold.
def is_pinching(hand_landmarks, finger_tip, pinch_threshold=0.75):
    index_tip = hand_landmarks.landmark[INDEX_TIP]
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_pip = hand_landmarks.landmark[6]  # Proximal phalanx of index finger
    index_len = distance_between_lms(index_tip, index_pip)
    pinch_len = distance_between_lms(hand_landmarks.landmark[finger_tip], thumb_tip)

    return pinch_len / index_len < pinch_threshold

# Determine if in drawing state
def is_drawing_state(hand_landmarks):
    return is_pinching(hand_landmarks, INDEX_TIP)

# Determine if in clearing state
def is_clearing_state(hand_landmarks):
    return is_pinching(hand_landmarks, MIDDLE_TIP)

# Get the position of the index tip
def get_index_tip(hand_landmarks):
    tip = hand_landmarks.landmark[INDEX_TIP]
    return (tip.x, tip.y)

# Calculate Euclidean distance between two points
def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def distance_between_lms(lm1, lm2):
    return distance_between_points((lm1.x, lm1.y), (lm2.x, lm2.y))

# Smooth a drawn stroke using Chaikin's algorithm.
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

# Smoothing using a simple moving average
def smooth_stroke_moving_average(points, window_size=5):
    if window_size < 3:
        raise ValueError("window_size must be at least 3")

    points_array = np.array(points)
    smoothed_points = np.copy(points_array)

    for i in range(1, len(points) - 1):
        start_index = max(i - window_size // 2, 0)
        end_index = min(i + window_size // 2 + 1, len(points))
        smoothed_points[i] = np.mean(points_array[start_index:end_index], axis=0)

    return [tuple(point) for point in smoothed_points]

# Draw strokes on the image
def draw_strokes(image, strokes, color=(255, 0, 0), thickness=5):
    h, w, _ = image.shape
    for stroke in strokes:
        stroke = chaikins_algorithm(stroke)  # Apply Chaikin's algorithm
        stroke = smooth_stroke_moving_average(stroke)  # Apply moving average

        for i in range(len(stroke) - 1):
            pt1 = (int(stroke[i][0] * w), int(stroke[i][1] * h))
            pt2 = (int(stroke[i + 1][0] * w), int(stroke[i + 1][1] * h))
            cv2.line(image, pt1, pt2, color, thickness)

def draw_point(image, landmark, color=(0, 255, 0), radius=5):
    h, w, _ = image.shape
    x, y = int(landmark.x * w), int(landmark.y * h)
    cv2.circle(image, (x, y), radius, color, -1)

def draw_center_box(image, color=(255, 0, 0), thickness=2):
    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 3
    width, height = w // 2, h // 3
    # width, height = int(w*0.7), int(h*0.5)

    top_left = (center_x - width // 2, center_y - height // 2)
    bottom_right = (center_x + width // 2, center_y + height // 2)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def save_strokes_txt(strokes, filename):
    with open(filename, 'w') as file:
        for i, stroke in enumerate(strokes):
            stroke_str = ' '.join(f'{x:.4f},{y:.4f}' for x, y in stroke)
            file.write(f'Stroke {i+1}: {stroke_str}\n')

def show_frame():
    global drawing
    global last_pos
    global current_stroke
    global cleared

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            drawing = False
            if not cleared:
                if is_clearing_state(hand_landmarks):
                    strokes.clear()
                    cleared = True

    draw_strokes(frame, strokes)
    draw_center_box(frame, color=(255, 0, 0)) 
    save_strokes_txt(strokes, 'strokes.txt')

    scale_percent = 200
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

def on_closing():
    global running
    running = False
    root.destroy()
    cap.release()
    hands.close()

root = tk.Tk()
lmain = tk.Label(root)
lmain.pack()

cap = cv2.VideoCapture(0)
running = True
show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
