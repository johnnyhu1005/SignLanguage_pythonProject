import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import socket

# 載入訓練好的雙手模型
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# 設定手勢數量及名稱
GESTURES = ["Thank you", "Thank you2", "Morning1", "Morning2", "Morning3", "Morning4", "Night", "welcome1", "welcome2",
            "Friend1", "Friend2", "Yes", "No1", "No2", "Treasure1", "Treasure2", "Treasure3", "Raining1", "Raining2",
            "Need1", "Need2", "Father1", "Father2", "Sunny1", "Sunny2", "Like1", "Like2", "Eat", "Drink1",
            "Drink2", "mom1", "mom2", "teacher1", "teacher2", "happy1", "happy2", "angry1","master1","master2"]

# 初始化 MediaPipe 和繪圖工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#設定目標 IP 和 Port (Unity 監聽的端口)
UDP_IP = "127.0.0.1"  # 本機 IP (可根據需求調整)
UDP_PORT = 5005        # Unity 使用的端口 (需和 Unity 端保持一致)

#建立 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 初始化變數
combo_triggered = False
combo_display_time = 0  # 控制顯示組合訊息的時間
detection_interval = 0.01  # 設置間隔時間 (秒)
tolerance_time = 2  # 容錯時間 (秒)
gesture_history = []  # 記錄最近偵測的手勢
error_tolerance = 50  # 容許的最大錯誤數量
current_errors = 0   # 當前錯誤計數

# 定義各個手勢順序組合
morning_sequence = ["Morning1", "Morning2", "Morning3", "Morning4"]  # "Morning" 手勢順序
goodnight_sequence = ["Night", "Morning3", "Morning4"]  # "GoodNight" 手勢順序
welcome_sequence = ["welcome1", "welcome2"]
friend_sequence = ["Friend1", "Friend2"]
no_sequence = ["No1", "No2"]
treasure_sequence = ["Treasure1", "Treasure2", "Treasure3"]
raining_sequence = ["Raining1", "Raining2"]
need_sequence = ["Need1", "Need2"]
father_sequence = ["Father1", "Father2"]
doctor_sequence = ["doctor1", "doctor2"]
mom_sequence = ["mom1", "mom2"]
teacher_sequence = ["teacher1", "teacher2"]
happy_sequence = ["happy1", "happy2"]
like_sequence = ["Like1", "Like2"]
sunny_sequence = ["Sunny1", "Sunny2"]
drink_sequence = ["Drink1", "Drink2"]
thanks_sequence = ["Thank you", "Thank you2"]
master_sequence = ["master1", "master2"]
eat_sequence = ["Eat", "Eat", "Eat", "Eat"]

# 定義當前的手勢步驟
current_morning_step = 0  # 當前的 "Morning" 手勢步驟
current_goodnight_step = 0  # 當前的 "GoodNight" 手勢步驟
current_welcome_step = 0
current_friend_step = 0
current_no_step = 0
current_treasure_step = 0
current_raining_step = 0
current_need_step = 0
current_father_step = 0
current_doctor_step = 0
current_mom_step = 0
current_teacher_step = 0
current_happy_step = 0
current_like_step = 0
current_sunny_step = 0
current_drink_step = 0
current_thanks_step = 0
current_master_step = 0
current_eat_step = 0

current_combo = None  # 當前顯示的手勢組合

# 初始化 MediaPipe Hands，允許偵測雙手
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.1, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # 轉換影像為 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 處理影像以偵測手勢
        results = hands.process(image)

        # 轉換回 BGR 以顯示影像
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

        # # 取得影像的寬度和高度
        # height, width = image.shape[:2]
        #
        # # 計算影像中心的位置
        # center_x = width // 2
        # center_y = height // 2
        #
        # # 定義橢圓的參數
        # ellipse_center = (center_x, center_y-50)  # 橢圓的中心
        # axes_length = (width // 6, height // 4)  # 橢圓的軸長 (寬度, 高度)
        # angle = 0  # 旋轉角度
        # start_angle = 0  # 起始角度
        # end_angle = 360  # 結束角度
        # color = (255, 0, 0)  # 綠色
        # thickness = 2  # 線條的厚度
        #
        # # 繪製橢圓形頭部輪廓
        # cv2.ellipse(image, ellipse_center, axes_length, angle, start_angle, end_angle, color, thickness)

        # 如果偵測到手勢，則進行辨識
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append([lm.x, lm.y, lm.z])
                landmarks.extend(np.array(hand_data).flatten())

            # 若只有一隻手，則補全另一隻手為 0
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)

            landmarks = np.array(landmarks)

            # 檢查 landmarks 長度是否為 126，若不足則補足；若多於126則截取
            if len(landmarks) != 126:
                landmarks = np.pad(landmarks, (0, max(0, 126 - len(landmarks))), 'constant')[:126]

            # 使用模型進行推理
            prediction = model.predict(np.expand_dims(landmarks, axis=0))
            predicted_gesture = GESTURES[np.argmax(prediction)]

            # 如果手勢偵測到某系列手勢，則檢查其順序
            if predicted_gesture == morning_sequence[current_morning_step]:
                current_morning_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_morning_step == len(morning_sequence):
                    current_combo = "Good Morning"
                    combo_display_time = current_time
                    current_morning_step = 0  # 重置步驟
            elif predicted_gesture == goodnight_sequence[current_goodnight_step]:
                current_goodnight_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_goodnight_step == len(goodnight_sequence):
                    current_combo = "Good Night"
                    combo_display_time = current_time
                    current_goodnight_step = 0  # 重置步驟
            elif predicted_gesture == welcome_sequence[current_welcome_step]:
                current_welcome_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_welcome_step == len(welcome_sequence):
                    current_combo = "You're Welcome"
                    combo_display_time = current_time
                    current_welcome_step = 0  # 重置步驟
            elif predicted_gesture == friend_sequence[current_friend_step]:
                current_friend_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_friend_step == len(friend_sequence):
                    current_combo = "Friend"
                    combo_display_time = current_time
                    current_friend_step = 0  # 重置步驟
            elif predicted_gesture == no_sequence[current_no_step]:
                current_no_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_no_step == len(no_sequence):
                    current_combo = "No"
                    combo_display_time = current_time
                    current_no_step = 0  # 重置步驟
            elif predicted_gesture == treasure_sequence[current_treasure_step]:
                current_treasure_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_treasure_step == len(treasure_sequence):
                    current_combo = "Treasure"
                    combo_display_time = current_time
                    current_treasure_step = 0  # 重置步驟
            elif predicted_gesture == raining_sequence[current_raining_step]:
                current_raining_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_raining_step == len(raining_sequence):
                    current_combo = "Raining"
                    combo_display_time = current_time
                    current_raining_step = 0  # 重置步驟
            elif predicted_gesture == need_sequence[current_need_step]:
                current_need_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_need_step == len(need_sequence):
                    current_combo = "Need"
                    combo_display_time = current_time
                    current_need_step = 0  # 重置步驟
            elif predicted_gesture == father_sequence[current_father_step]:
                current_father_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_father_step == len(father_sequence):
                    current_combo = "Father"
                    combo_display_time = current_time
                    current_father_step = 0  # 重置步驟
            elif predicted_gesture == doctor_sequence[current_doctor_step]:
                current_doctor_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_doctor_step == len(doctor_sequence):
                    current_combo = "Doctor"
                    combo_display_time = current_time
                    current_doctor_step = 0  # 重置步驟
            elif predicted_gesture == mom_sequence[current_mom_step]:
                current_mom_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_mom_step == len(mom_sequence):
                    current_combo = "Mom"
                    combo_display_time = current_time
                    current_mom_step = 0  # 重置步驟
            elif predicted_gesture == teacher_sequence[current_teacher_step]:
                current_teacher_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_teacher_step == len(teacher_sequence):
                    current_combo = "Teacher"
                    combo_display_time = current_time
                    current_teacher_step = 0  # 重置步驟
            elif predicted_gesture == happy_sequence[current_happy_step]:
                current_happy_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_happy_step == len(happy_sequence):
                    current_combo = "Happy"
                    combo_display_time = current_time
                    current_happy_step = 0  # 重置步驟
            elif predicted_gesture == like_sequence[current_like_step]:
                current_like_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_like_step == len(like_sequence):
                    current_combo = "Like"
                    combo_display_time = current_time
                    current_like_step = 0  # 重置步驟
            elif predicted_gesture == sunny_sequence[current_sunny_step]:
                current_sunny_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_sunny_step == len(sunny_sequence):
                    current_combo = "Sunny"
                    combo_display_time = current_time
                    current_sunny_step = 0  # 重置步驟
            elif predicted_gesture == drink_sequence[current_drink_step]:
                current_drink_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_drink_step == len(drink_sequence):
                    current_combo = "Drink"
                    combo_display_time = current_time
                    current_drink_step = 0  # 重置步驟
            elif predicted_gesture == thanks_sequence[current_thanks_step]:
                current_thanks_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_thanks_step == len(thanks_sequence):
                    current_combo = "Thanks"
                    combo_display_time = current_time
                    current_thanks_step = 0  # 重置步驟
            elif predicted_gesture == master_sequence[current_master_step]:
                current_master_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_master_step == len(master_sequence):
                    current_combo = "Master"
                    combo_display_time = current_time
                    current_master_step = 0  # 重置步驟
            elif predicted_gesture == eat_sequence[current_eat_step]:
                current_eat_step += 1  # 如果順序正確，進入下一個步驟
                current_errors = 0  # 重置錯誤計數
                if current_eat_step == len(eat_sequence):
                    current_combo = "Eat"
                    combo_display_time = current_time
                    current_eat_step = 0  # 重置步驟

            else:
                # 偵測到錯誤動作，增加錯誤計數，超過容錯範圍時重置
                current_errors += 1
                if current_errors > error_tolerance:
                    # 重置手勢序列
                    current_morning_step = 0  # 重置 "Morning" 手勢序列
                    current_goodnight_step = 0  # 重置 "GoodNight" 手勢序列
                    current_welcome_step = 0  # 重置 "Welcome" 手勢序列
                    current_friend_step = 0
                    current_no_step = 0
                    current_treasure_step = 0
                    current_raining_step = 0
                    current_need_step = 0
                    current_father_step = 0
                    current_doctor_step = 0
                    current_mom_step = 0
                    current_teacher_step = 0
                    current_happy_step = 0
                    current_errors = 0  # 重置錯誤計數

            # 畫出預測結果
            cv2.putText(image, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(image, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # 如果組合動作觸發且未超過顯示時間，顯示該組合訊息
            if current_combo and current_time - combo_display_time <= 1:
                display_text = f"Motion: {current_combo}"
                # 傳送訊息
                sock.sendto(current_combo.encode(), (UDP_IP, UDP_PORT))
                print(f"Sent message: {current_combo}")
                cv2.putText(image, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(image, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif current_time - combo_display_time > 1:
                current_combo = None  # 重置動作組合

            # 畫出手勢 landmarks 及參考線
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 顯示影像
        cv2.imshow('Hand Gesture Recognition', image)

        # 按下 q 鍵退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
