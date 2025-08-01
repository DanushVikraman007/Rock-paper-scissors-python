import cv2

import mediapipe as mp
import random
import time
from collections import deque, Counter
import statistics as st

# === Constants ===
MOVES = ["Rock", "Paper", "Scissors"]
DISPLAY_VALUES = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]
STATE_WAIT = "WAIT"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_SHOW_RESULT = "RESULT"

# === MediaPipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# === Game Variables ===
cpu_score = 0
player_score = 0
game_state = STATE_WAIT
countdown_start = 0
round_result_shown_time = 0
round_duration = 2  # seconds to show result
countdown_duration = 3  # seconds

# === Helper Functions ===
def calculate_winner(cpu, player):
    if player == "Invalid":
        return "Invalid!"
    if player == cpu:
        return "Tie!"
    if (player == "Rock" and cpu == "Scissors") or \
       (player == "Paper" and cpu == "Rock") or \
       (player == "Scissors" and cpu == "Paper"):
        return "You win!"
    return "CPU wins!"

def detect_fingers(hand_landmarks, label):
    count = 0
    # Index, middle, ring, pinky
    if hand_landmarks[8][2] < hand_landmarks[6][2]: count += 1
    if hand_landmarks[12][2] < hand_landmarks[10][2]: count += 1
    if hand_landmarks[16][2] < hand_landmarks[14][2]: count += 1
    if hand_landmarks[20][2] < hand_landmarks[18][2]: count += 1
    # Thumb (left vs right)
    if label == "Left" and hand_landmarks[4][1] > hand_landmarks[3][1]:
        count += 1
    elif label == "Right" and hand_landmarks[4][1] < hand_landmarks[3][1]:
        count += 1
    return count

def get_player_move(finger_count):
    if finger_count <= 5:
        return DISPLAY_VALUES[finger_count]
    return "Invalid"

def draw_ui(image, player_choice, cpu_choice, winner, player_score, cpu_score):
    cv2.putText(image, "You", (90, 75), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)
    cv2.putText(image, "CPU", (1050, 75), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
    cv2.putText(image, player_choice, (45, 375), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)
    cv2.putText(image, cpu_choice, (1000, 375), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
    cv2.putText(image, winner, (530, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 5)
    cv2.putText(image, str(player_score), (145, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)
    cv2.putText(image, str(cpu_score), (1100, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

# === Main ===
webcam = cv2.VideoCapture(0)
move_buffer = deque(["Nothing"] * 5, maxlen=5)
cpu_choice = "Nothing"
player_choice = "Nothing"
winner = "None"

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Camera error.")
            break

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        count = 0
        detected_move = "Nothing"

        if results.multi_hand_landmarks:
            for hand_index, hand_landmark in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)

                label = results.multi_handedness[hand_index].classification[0].label
                landmarks = []
                h, w, _ = image.shape

                for id, lm in enumerate(hand_landmark.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, x, y])

                if len(landmarks) >= 21:
                    count = detect_fingers(landmarks, label)
                    detected_move = get_player_move(count)

        move_buffer.appendleft(detected_move)

        # Get stable player move
        try:
            player_choice = st.mode(move_buffer)
        except st.StatisticsError:
            player_choice = "Nothing"

        # === Game Logic ===
        if game_state == STATE_WAIT:
            cv2.putText(image, "Get Ready! Show your move...", (350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            countdown_start = time.time()
            game_state = STATE_COUNTDOWN

        elif game_state == STATE_COUNTDOWN:
            elapsed = int(time.time() - countdown_start)
            remaining = countdown_duration - elapsed
            if remaining > 0:
                cv2.putText(image, f"{remaining}", (600, 400),
                            cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 255), 8)
            else:
                cpu_choice = random.choice(MOVES)
                winner = calculate_winner(cpu_choice, player_choice)
                if winner == "You win!":
                    player_score += 1
                elif winner == "CPU wins!":
                    cpu_score += 1
                round_result_shown_time = time.time()
                game_state = STATE_SHOW_RESULT

        elif game_state == STATE_SHOW_RESULT:
            if time.time() - round_result_shown_time > round_duration:
                game_state = STATE_WAIT  # Start new round

        draw_ui(image, player_choice, cpu_choice, winner, player_score, cpu_score)
        cv2.imshow("Rock Paper Scissors", image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

webcam.release()
cv2.destroyAllWindows()
