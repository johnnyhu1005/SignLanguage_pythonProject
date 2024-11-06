import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 載入數據資料夾
DATA_DIR = "data"
# 設定手勢數量及名稱
GESTURES = ["Thank you", "Thank you2", "Bye", "Morning1", "Morning2", "Morning3", "Morning4", "Night", "welcome1", "welcome2",
            "Friend1", "Friend2", "Yes", "No1", "No2", "Treasure1", "Treasure2", "Treasure3", "Raining1", "Raining2",
            "Need1", "Need2", "Father1", "Father2", "Sunny1", "Sunny2", "Like1", "Like2", "Eat", "Drink1",
            "Drink2", "doctor1", "doctor2", "mom1", "mom2", "teacher1", "teacher2","happy1", "happy2", "angry1","master1","master2"]

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

# 分割訓練和測試數據
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立更深的 DNN 模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(126,)),  # 21個點 x 3個座標 x 2隻手
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # 添加 Dropout 層以防止過擬合
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 保存模型
model.save('gesture_recognition_model.h5')
print("手勢模型已保存")

# 繪製訓練和驗證的損失與準確度曲線
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# 預測測試集並生成混淆矩陣
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GESTURES, yticklabels=GESTURES)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
