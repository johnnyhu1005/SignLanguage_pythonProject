import cv2
import numpy as np
import mediapipe as mp
import os

# 初始化 MediaPipe 和繪圖工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 設定手勢數量及名稱
GESTURES = ["Thank you", "Thank you2", "Bye", "Morning1", "Morning2", "Morning3", "Morning4", "Night","welcome1", "welcome2",
            "Friend1", "Friend2", "Yes","No1","No2"]  # 請替換為實際手語名稱

# 建立資料夾以保存訓練數據"
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 轉換影像為 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 處理影像以偵測雙手或單手
        results = hands.process(image)

        # 轉換回 BGR 以顯示影像
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果偵測到手勢，則畫出標記和參考線
        if results.multi_hand_landmarks:
            landmarks = []

            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append([lm.x, lm.y, lm.z])
                landmarks.extend(np.array(hand_data).flatten())

                # 繪製手部 landmarks 和連接線
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 顯示 landmarks 並等待鍵盤輸入標記
            cv2.imshow('Hand Gesture Capture', image)

        else:
            # 即使沒有偵測到手部，也顯示影像並繼續
            cv2.imshow('Hand Gesture Capture', image)

        # 按下指定鍵以擷取手勢資料
        key = cv2.waitKey(10)
        if key == ord('q'):  # 按下 q 鍵退出
            break
        elif key == ord('c'):  # 按下 c 鍵擷取當前手勢
            # 讓用戶選擇手勢標籤
            print("請輸入對應的手勢編號：")
            label = int(input())
            if 0 <= label < len(GESTURES):
                gesture_dir = os.path.join(DATA_DIR, GESTURES[label])
                if not os.path.exists(gesture_dir):
                    os.makedirs(gesture_dir)
                np.save(os.path.join(gesture_dir, f"{len(os.listdir(gesture_dir))}.npy"), landmarks)
                print(f"已擷取手勢: {GESTURES[label]}")

    cap.release()
    cv2.destroyAllWindows()
