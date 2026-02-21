import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

# Pastel Colors (BGR)
colors = [
    (255, 182, 193),  # Pink
    (255, 221, 148),  # Peach
    (186, 225, 255),  # Baby Blue
    (204, 255, 204),  # Mint
    (230, 230, 250)   # Lavender
]

draw_color = colors[0]
brush_thickness = 5
eraser_thickness = 40


def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def draw_animated_cursor(frame, x, y):
    # Outer glow rings
    for i in range(20, 0, -5):
        cv2.circle(frame, (x, y), i, (255, 200, 255), 1)

    # Inner pink circle
    cv2.circle(frame, (x, y), 6, (255, 105, 180), -1)

    # Sparkles
    sparkle_positions = [
        (x + 15, y - 15),
        (x - 15, y + 10),
        (x + 10, y + 15)
    ]

    for sx, sy in sparkle_positions:
        cv2.circle(frame, (sx, sy), 2, (255, 255, 0), -1)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.ones_like(frame) * 255  # White background

    h, w, c = frame.shape

    # Aesthetic top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (245, 245, 245), -1)

    x_start = 30
    for color in colors:
        cv2.rectangle(frame, (x_start, 25), (x_start + 60, 75), color, -1)
        x_start += 90

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            draw_animated_cursor(frame, x, y)

            finger_state = fingers_up(hand_landmarks)
            total_fingers = sum(finger_state)

            # ✋ 5 Fingers = Erase
            if total_fingers == 5:

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         (255, 255, 255), eraser_thickness)

                prev_x, prev_y = x, y

            # ☝️ Only index finger = Draw
            elif finger_state[1] == 1 and total_fingers == 1:

                # Select color
                if y < 90:
                    prev_x, prev_y = 0, 0

                    box_index = (x - 30) // 90
                    if 0 <= box_index < len(colors):
                        draw_color = colors[box_index]

                else:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y

                    cv2.line(canvas, (prev_x, prev_y), (x, y),
                             draw_color, brush_thickness)

                    prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

    # Combine canvas with frame (so cursor shows on top)
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    final_output = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Cute Aesthetic Air Canvas 💖✨", final_output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.ones_like(frame) * 255
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()