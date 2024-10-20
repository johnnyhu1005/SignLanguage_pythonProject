import os
import numpy as np
import tensorflow as tf

# 載入數據資料夾
DATA_DIR = "data"
GESTURES = ["Thank you", "Thank you2", "Bye", "Morning1", "Morning2", "Morning3", "Morning4", "Night",
            "welcome1", "welcome2", "Friend1", "Friend2", "Yes", "No1", "No2"]  # 與 main.py 中相同

X, y = [], []

# 從每個手勢資料夾載入數據
for idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(gesture_dir):
        data = np.load(os.path.join(gesture_dir, file))

        # 如果數據是單手手勢，補上 63 個 0 作為另一隻手的數據
        if len(data) == 63:  # 單手資料
            data = np.concatenate([data, np.zeros(63)])  # 補上 63 個 0
        X.append(data)
        y.append(idx)

# 將數據轉換為 NumPy 格式
X = np.array(X)
y = np.array(y)

# 建立 TensorFlow 模型，輸入維度固定為 126（21 個手勢點，3 個坐標，2 隻手的數據）
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(126,)),  # 21個點 x 3個座標 x 2隻手
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X, y, epochs=50)

# 保存模型
model.save('gesture_recognition_model.h5')
print("手勢模型已保存")
