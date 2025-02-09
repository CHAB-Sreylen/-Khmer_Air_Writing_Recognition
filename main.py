import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk,ImageFont,ImageDraw
import threading
import time 
import numpy as np
import pandas as pd 
import os
import csv
from predict import predict_char

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Define important fingertip landmarks
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False,max_num_hands = 2,
                       min_detection_confidence=0.5
                        ,min_tracking_confidence =0.5)
mp_drawing = mp.solutions.drawing_utils


strokes = []
predicted_char = []
drawing = False
cleared = True
move_threshold = 0.01
save_threshold = 0.1

show_saved_message=False
saved_message_time = 0


# define if the specified fingertip is pinching the thumb.
def is_pinching(hand_landmarks, finger_tip, pinch_threshold=0.5):
    index_tip = hand_landmarks.landmark[8]
    thump_tip = hand_landmarks.landmark[4]
    index_pip = hand_landmarks.landmark[6]
    index_len = distance_between_lms(index_tip, index_pip)
    pinch_len = distance_between_lms(hand_landmarks.landmark[finger_tip], 
                                     thump_tip)
    return pinch_len / index_len < pinch_threshold

# 1. Checking if the hand is a drawing state 
def is_drawing_state(hand_landmarks):
    return is_pinching(hand_landmarks,INDEX_TIP)

# 2. Checking if the hand is a saving state 
def is_saving_state(hand_landmarks):
    # Define the landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if all fingertips are near their respective MCP joints (fingers curled)
    thumb_curled = abs(thumb_tip.x - thumb_mcp.x) < 0.05
    index_curled = abs(index_tip.y - index_mcp.y) < 0.05
    middle_curled = abs(middle_tip.y - middle_mcp.y) < 0.05
    ring_curled = abs(ring_tip.y - ring_mcp.y) < 0.05
    pinky_curled = abs(pinky_tip.y - pinky_mcp.y) < 0.05

    return thumb_curled and index_curled and middle_curled and ring_curled and pinky_curled


 # Add conditions for ring and pinky
def get_index_tip(hand_landmarks):
    tip = hand_landmarks.landmark[8]
    return (tip.x, tip.y)


# we calculate the Euclidean distance between two points
def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# we calculates the distance between two MediaPipe landmarks.
def distance_between_lms(lm1, lm2):
    return distance_between_points((lm1.x, lm1.y), (lm2.x, lm2.y))


# Using chaikin algorithm as a corner-cutting algorithm used for curve smoothing.
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

# A method for moving average filter to smooth a sequence of 2D points
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

# Draw a given strokes on image  
def draw_strokes(image, strokes, color=(255, 255, 255), thickness=5):
    h, w, _ = image.shape
    for stroke in strokes:
        stroke = smooth_stroke_moving_average(stroke)
        for i in range(len(stroke) - 1):
            pt1 = (int(stroke[i][0] * w), int(stroke[i][1] * h))
            pt2 = (int(stroke[i + 1][0] * w), int(stroke[i + 1][1] * h))
            cv2.line(image, pt1, pt2, color, thickness)


# Draw a point on the image at the given landmark location 
def draw_point(image, landmark, color=(0, 255, 0), radius=5):
    h, w, _ = image.shape
    x, y = int(landmark.x * w), int(landmark.y * h)
    cv2.circle(image, (x, y), radius, color, -1)

def draw_center_box(image, overlay_img1, overlay_img2, khmer_text, font_khmer, predicted_text, khmer_text1, text_position, text_color, color=(112, 97, 200), thickness=2):
    h, w, _ = image.shape
    padding = 20
    box_width, box_height = 480, 240
    text_box_color = (112,97,200)
    top_left_x = padding
    top_left_y = padding
    bottom_right_x = top_left_x + box_width
    bottom_right_y = top_left_y + box_height

    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

    overlay_size = (100, 110)  
    overlay_img1 = cv2.resize(overlay_img1, overlay_size)
    overlay_img2 = cv2.resize(overlay_img2, overlay_size)

    image_top_left1 = (bottom_right_x + padding, top_left_y)
    image_top_left2 = (image_top_left1[0], image_top_left1[1] + 110 + padding)

    for overlay_img, position in zip([overlay_img1, overlay_img2], 
                                      [image_top_left1, image_top_left2]):
        if position[0] + 100 > w:
            position = (w - 100, position[1])
        if position[1] + 110 > h:
            position = (position[0], h - 110)

        overlay_region = image[position[1]:position[1] + 110, position[0]:position[0] + 100]
        blended = cv2.addWeighted(overlay_img, 1.0, overlay_region, 0.0, 0)
        image[position[1]:position[1] + 110, position[0]:position[0] + 100] = blended

    img_pil = Image.fromarray(image)

    draw = ImageDraw.Draw(img_pil)
    left_text_position = (30, (h // 2) + 60)
    # draw.text(left_text_position, khmer_text, font=font_khmer, fill=text_color)

    draw = ImageDraw.Draw(img_pil)
    right_text_position = (100, (h // 2) + 60)
    # draw.text(right_text_position, khmer_text1, font=font_khmer, fill=text_color)

    return np.array(img_pil)

# Convert a point of stroke into the input format in prediction model 
def formatinput(points):
    formatted_data = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
            formatted_data.append(f'{x:.6f},{y:.6f}')

    return ', '.join(formatted_data)

def flatten_list(nested_list):
    """Flatten a nested list of tuples."""
    return [item for sublist in nested_list for item in sublist]

strokes_for_saving = []

fontpath = "fonts/KhmerOS.ttf"
predicted_text = []
data_folder = '/data'
os.makedirs(data_folder, exist_ok=True)
strokes_for_saving = []


def save_strokes_csv(strokes, filename):
    combined_stroke = []
    for stroke in strokes:
        combined_stroke.extend(stroke)  
    formatted_data = []
    for point in combined_stroke:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
            formatted_data.append(f'{x:.6f},{y:.6f}')
        else:
            print(f"Warning: Skipping invalid point {point}")
    file_path = os.path.join(data_folder, filename)
    with open(file_path, 'a', newline='') as file:
        if formatted_data:  
            file.write(','.join(formatted_data) + '\n')
        else:
            file.write('\n')


# calculate the bounding box of given stroke 
def calculate_bounding_box(strokes, image_shape):
    all_points = [point for stroke in strokes for point in stroke]
    
    if not all_points:
        return None
    
    x_coordinates = [int(point[0] * image_shape[1]) for point in all_points]
    y_coordinates = [int(point[1] * image_shape[0]) for point in all_points]
    
    xmin = min(x_coordinates)
    xmax = max(x_coordinates)
    ymin = min(y_coordinates)
    ymax = max(y_coordinates) 
    
    return (xmin, ymin, xmax, ymax)
predicted_text_radius = []  
DRAWING_SPEED = 5

# Apply circular reveal on the text 
def apply_circular_reveal(char_surface, pil_image_size, x, y, char,
                          font, font_size, current_radius):
    max_radius = int(1.5 * font_size)
    mask = Image.new("L", pil_image_size, 0)
    mask_draw = ImageDraw.Draw(mask)
    char_draw = ImageDraw.Draw(char_surface)
    bbox = char_draw.textbbox((x, y), char, font=font)
    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2
    print(f"Character {char} Center: ({center_x}, {center_y})")
    mask_draw.ellipse(
        (center_x - current_radius, center_y - current_radius,
         center_x + current_radius, center_y + current_radius),
        fill=255
    )
    print(f"Radius: {current_radius}, Center: ({center_x}, {center_y})")
    char_surface_with_mask = Image.composite(char_surface, 
                             Image.new("RGBA", pil_image_size, (0, 0, 0, 0)), mask)
    return char_surface_with_mask, bbox


# Display text on the frame 
def display_predicted_text_on_frame(frame, predicted_text, bounding_box, fontpath="fonts/KhmerOS.ttf", text_color=(0, 0, 0)):
    """Main function to display predicted text with circular reveal on the frame."""
    global predicted_text_radius
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    (xmin, ymin, xmax, ymax) = bounding_box
    bbox_height = ymax - ymin
    
    font_size = max(int(0.8 * bbox_height),10)
    try:
        font = ImageFont.truetype(fontpath, font_size)
    except Exception as e:
        print(f"Error loading font: {e}. Using default font.")
        font = ImageFont.load_default()

    if not predicted_text:
        return frame,True 

    if len(predicted_text_radius) < len(predicted_text ):
        predicted_text_radius.extend([0] * (len(predicted_text) - len(predicted_text_radius)))

    start_x = xmax + 30
    start_y = ymax - bbox_height - 30 
    previous_char_width = 0  

    combined_surface = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    animation_complete = True 
    for i, char in enumerate(predicted_text):
        x = start_x + previous_char_width + (3 if i > 0 else 0)
        y = start_y

        char_surface = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_surface)
        char_draw.text((x, y), char, font=font, fill=text_color)
        current_radius = predicted_text_radius[i]
        max_radius = int(1.5 * font_size)
        if current_radius < max_radius:
            current_radius += DRAWING_SPEED
            predicted_text_radius[i] = current_radius
            animation_complete = False
        else:
            current_radius = max_radius
        char_surface_with_mask, bbox = apply_circular_reveal(char_surface, pil_image.size, x, y, char, font, font_size, current_radius)
        combined_surface = Image.alpha_composite(combined_surface, char_surface_with_mask)
        previous_char_width = bbox[2] - bbox[0]
    
    pil_image = Image.alpha_composite(pil_image.convert("RGBA"), combined_surface)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return frame,animation_complete


# Resets the predicted text radius 
def reset_predicted_text_radius(new_predicted_text):
    """Reset the radius values for the circular reveal when a new prediction is added."""
    global predicted_text_radius
    predicted_text_radius = [0] * len(new_predicted_text)

def text_box(frame, top_left_x, top_left_y, box_width, box_height, offset_y=20, 
            height=180, text_box_width=480, text_color=(255, 0, 0), 
            fontpath="fonts/KhmerOS.ttf", predicted_text=[]):
    top_left_y = top_left_y + box_height + offset_y
    bottom_right_x = top_left_x + text_box_width
    bottom_right_y = top_left_y + height
    cv2.rectangle(frame, (top_left_x, top_left_y), 
                  (bottom_right_x, bottom_right_y), (112, 97, 200), 2)
    return frame


def show_frame():
    global drawing
    global last_pos
    global current_stroke
    global cleared
    global show_saved_message
    global saved_message_time
    global predicted_char
    global predicted_text
    global bounding_box
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fontpath="fonts/KhmerOS.ttf"
    # Process frame for hand landmarks.
    hand_results = hands.process(frame)
    
    padding = 20 
    box_width,box_height = 400,240 
    top_left_x = padding
    top_left_y = padding
    bottom_right_x = top_left_x + box_width
    bottom_right_y = top_left_y + box_height

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[-1]
        
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
                # predicted_text.clear()

            if (
                top_left_x <=tip_pos[0] * frame.shape[1] <= bottom_right_x
                and top_left_y<= tip_pos[1] * frame.shape[0] <=bottom_right_y
            ):
                if distance_between_points(tip_pos, last_pos)> move_threshold:
                    current_stroke.append(tip_pos)
                    last_pos = tip_pos
        else:
            
            drawing = False
            if is_saving_state(hand_landmarks):
                    if strokes_for_saving and any(stroke for stroke in strokes_for_saving if stroke):
                        print (strokes_for_saving)
                        flattened_strokes = flatten_list(strokes_for_saving)
                        formatted_strokes = formatinput(flattened_strokes)
                        print("Formatted strokes for saving:", formatted_strokes)
                        predicted_char = predict_char(formatted_strokes)
                        # predicted_char = "ក"
                        # save_strokes_csv(strokes_for_saving, filename='save_strokes.csv')
                        save_strokes_csv(formatted_strokes, filename='save_strokes.csv')
                        print(predicted_char)
                        bounding_box = calculate_bounding_box(strokes_for_saving,frame.shape)
                        # if bounding_box:
                        print("Bounding Box:",bounding_box)
                        if predicted_char:
                            reset_predicted_text_radius([predicted_char])
                            # predicted_text.clear()
                            predicted_text.append(predicted_char)
                            print("Updated predicted_text:", predicted_text)
                            predicted_char = ""
                        # frame = display_predicted_text_on_frame(frame, predicted_text,bounding_box) 
                        strokes_for_saving.clear()
                        show_saved_message = True
                        saved_message_time = time.time()
                        strokes.clear()
                        
    if predicted_text and bounding_box:
        current_predicted_text = [predicted_text[-1]] if predicted_text else []
        print(f"current predicted{current_predicted_text}")
    # if animation_complete
        frame,animation_complete = display_predicted_text_on_frame(frame,current_predicted_text,bounding_box) 
        if predicted_text: 
            
            try: 
                font_size = 40
                font = ImageFont.truetype(fontpath,font_size)
            except Exception as e:
                print(f"Error loading font: {e}")
                font = ImageFont.load_default()
        
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            # text_position = (40, 280)
            start_x, start_y = 40, 280
            line_spacing =10
            for i, text in enumerate(predicted_text):
                text_position=(start_x,start_y)
                draw.text(text_position, ''.join(predicted_text), font=font, fill=(255,255, 255))  # Adjust color as needed
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # print(f"Displayed predicted_text at position {text_position}: {predicted_text}")
            print(f"Displayed predicted_texts starting at position ({start_x}, {start_y}): {predicted_text}")
    if show_saved_message and time.time() - saved_message_time < 1:
        cv2.putText(frame,"Saved",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(31,33,71),2,cv2.LINE_AA)
    elif time.time() - saved_message_time >=1:
        show_saved_message = False
    
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)
    draw_strokes(frame, strokes)

    overlay_img1 = cv2.imread("image/pinching_index.png")
    overlay_img1 = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2RGB)
    overlay_img2 = cv2.imread("image/fisting.png")
    overlay_img2 = cv2.cvtColor(overlay_img2, cv2.COLOR_BGR2RGB)
 
    khmer_text = "sMeNr"
    khmer_text1 =  ''.join(predicted_char)
    fontpath = "fonts/KhmerOS.ttf"
    font_khmer = ImageFont.truetype(fontpath,70)
    text_color = (255, 0, 0)
    text_position = (10,10)
    
    text_box(frame, top_left_x, top_left_y,box_width, box_height,predicted_text = predicted_text)
    
    center_box_frame = draw_center_box(frame, overlay_img1, overlay_img2, khmer_text, font_khmer,khmer_text1, (10, 10), text_color,text_position)
    frame_resized = cv2.resize(frame, (root.winfo_screenwidth(), root.winfo_screenheight()))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

def exit_fullscreen(event=None):
    root.attributes('-fullscreen', False)
def on_closing():
    global running
    running = False
    root.destroy()
    cap.release()
    hands.close()

root = tk.Tk()     
root.attributes('-fullscreen', True)  
lmain = tk.Label(root)

lmain.pack(expand=True, fill='both')  

running = True

show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.bind("<Escape>",exit_fullscreen)
root.mainloop()

