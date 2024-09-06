import cv2
from PIL import Image, ImageDraw, ImageFont
from utils import smooth_stroke_moving_average
import numpy as np 
def draw_strokes(image, strokes, color=(255, 255, 255), thickness=5):
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

# def draw_center_box(image, overlay_img1, overlay_img2, khmer_text, font_khmer, text_position, text_color, color=(112,97,200), thickness=2):
#     h, w, _ = image.shape
#     padding = 20
#     text_box_padding = 20
#     box_width, box_height = 480, 240
#     text_box_height = 180
#     text_box_color = (112,97,200)
#     top_left_x = padding
#     top_left_y = padding
#     bottom_right_x = w - padding
#     bottom_right_y = h - padding

#     if top_left_x + box_width > bottom_right_x:
#         box_width = bottom_right_x - top_left_x
#     if top_left_y + box_height > bottom_right_y:
#         box_height = bottom_right_y - top_left_y

#     top_left = (top_left_x, top_left_y)
#     bottom_right = (top_left_x + box_width, top_left_y + box_height)
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.rectangle(mask, top_left, bottom_right, 255, -1)
#     image_with_box = cv2.bitwise_and(image, image, mask=mask)
#     cv2.rectangle(image_with_box, top_left, bottom_right, color, thickness)
#     text_box_top_left = (top_left[0], bottom_right[1] + text_box_padding)
#     text_box_bottom_right = (bottom_right[0], text_box_top_left[1] + text_box_height)
#     if text_box_bottom_right[1] > h:
#         text_box_bottom_right = (text_box_bottom_right[0], h)
#     cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, -1)
#     cv2.rectangle(image_with_box, text_box_top_left, text_box_bottom_right, text_box_color, thickness)
#     overlay_img1 = cv2.resize(overlay_img1, (100, 110))
#     overlay_img2 = cv2.resize(overlay_img2, (100, 110))
#     image_top_left1 = (bottom_right[0] + padding, top_left[1])
#     if image_top_left1[0] + 100 > w:
#         image_top_left1 = (w - 100, image_top_left1[1])
#     if image_top_left1[1] + 110 > h:
#         image_top_left1 = (image_top_left1[0], h - 110)
#     overlay_region1 = image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100]
#     blended1 = cv2.addWeighted(overlay_img1, 1.0, overlay_region1, 0.0, 0)
#     image_with_box[image_top_left1[1]:image_top_left1[1] + 110, image_top_left1[0]:image_top_left1[0] + 100] = blended1

#     image_top_left2 = (image_top_left1[0], image_top_left1[1] + 110 + 20)
#     if image_top_left2[0] + 100 > w:
#         image_top_left2 = (w - 100, image_top_left2[1])
#     if image_top_left2[1] + 110 > h:
#         image_top_left2 = (image_top_left2[0], h - 110)
#     overlay_region2 = image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100]
#     blended2 = cv2.addWeighted(overlay_img2, 1.0, overlay_region2, 0.0, 0)
#     image_with_box[image_top_left2[1]:image_top_left2[1] + 110, image_top_left2[0]:image_top_left2[0] + 100] = blended2

#     img_pil = Image.fromarray(image_with_box)
#     draw = ImageDraw.Draw(img_pil)
#     left_text_position = (30, (h // 2) +60)
#     draw.text(left_text_position, khmer_text, font=font_khmer, fill=text_color)
#     return np.array(img_pil)

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
