import os
import numpy as np
import tensorflow as tf

# 載入數據資料夾
DATA_DIR = "data"
GESTURES = ["Thank you"]  # 與 main.py 中相同

X, y = [], []

# 從每個手勢資料夾載入數據
for idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(gesture_dir):
        data = np.load(os.path.join(gesture_dir, file))
        X.append(data)
        y.append(idx)

# 將數據轉換為 NumPy 格式
X = np.array(X)
y = np.array(y)