# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:33:10 2024

@author: BTG
"""



# 필요한 라이브러리와 모듈을 가져옵니다.
from tqdm import tqdm
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# load_img : 이미지 파일을 로드
# img_to_array : 이미지를 배열로 변환하는 데 사용

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
# Sequential은 케라스에서 순차적 레이어를 쌓아 모델을 만드는 클래스
# 층(layer)을 순차적으로 쌓아 모델을 구성하는 클래스

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# 여기서는 여러 케라스 레이어를 가져옴. 이들은 신경망을 구성하는 데 사용

from tensorflow.keras.optimizers import Adam
# Adam은 최적화 알고리즘으로, 모델의 학습 과정을 관리
# 학습 과정에서 모델의 가중치를 업데이트하는 방법을 정의하는 최적화 알고리즘

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, AUC


import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import Callback
# Callback은 케라스에서 모델의 학습 과정 중 특정 이벤트가 발생했을 때 호출되는 함수들의 기본 클래스
# 학습 과정 중에 사용자 정의 동작을 구현하는 데 사용되는 클래스


"""
# 이름 바꾸기
# =============================================================================
# 20210106134858-1_10019.png
# 20210106134916-1_10020.png
# 20210106134933-0_10021.png
# 20210106134951-1_1.png
# 20210106135009-0_2.png
# 
# 이렇거든? 혹시 이 이미지들의 이름을 바꿀 수 있을까?
# 
# 20210106134916-1.png
# 20210106134933-0.png
# 20210106134951-1.png
# 20210106135009-0.png
# =============================================================================


def rename_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 파일 이름에서 첫 번째 '_' 이후의 부분을 제거합니다.
            parts = filename.split('_')
            new_name = parts[0] + '.' + filename.split('.')[-1]  # 확장자도 유지합니다.
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            
            # 파일 이름 변경
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_name}'")

# 함수 호출
directory = "C:/Users/USER/Desktop/dxf2png"  # 이 경로는 실제 경로로 변경해야 합니다.
rename_images(directory)
"""






# 이미지 데이터를 로드하고 전처리하는 함수를 정의합니다.
def load_data(directory, target_size=(224, 224)):
    images = []
    labels = []
    # 로드된 이미지와 레이블을 저장할 빈 리스트를 생성

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.jpg') or filename.endswith('.png'): # 이미지 파일(.jpg, .png)만을 처리
            img_path = os.path.join(directory, filename) # 각 이미지 파일의 전체 경로를 생성
            img = load_img(img_path, color_mode='grayscale', target_size=target_size) # 각 이미지 파일을 로드. target_size는 이미지의 크기를 조정
            img_array = img_to_array(img) # img_to_array: 로드된 이미지를 numpy 배열로 변환합니다.
            images.append(img_array)
            label = 1 if filename.endswith('-1.jpg') or filename.endswith('-1.png') else 0
            # : 파일 이름을 기반으로 레이블을 지정. 예를 들어, 파일 이름 끝에 '1'이 있으면 레이블을 1로, 그렇지 않으면 0으로 설정
            labels.append(label)

    return np.array(images), np.array(labels)


# 로드 데이터 함수를 호출하여 이미지와 레이블을 로드합니다.
images, labels = load_data("C:/Users/USER/Desktop/Proj. PCB/5. 형상복원 이미지/결과물/4) dxf_to_jpg_matching_label")
# =============================================================================
# # images2, labels2 = load_data("C:/Users/skyga/OneDrive/바탕 화면/PCB/4. 기존 이미지 핸들링/결과물/3) labeled image")
# 
# 
# 
# A = (labels == labels2)
# 
# num_true = np.count_nonzero(A)
# num_false = A.size - np.count_nonzero(A)
# print(num_true + num_false)
# 
# =============================================================================







# 데이터를 훈련 및 테스트 세트로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN 모델을 정의하고 구축하는 함수입니다.
# 모델 구축 함수 수정
def build_model(input_shape):  # 'input_shape 매개변수는 모델의 입력 형태를 결정합니다.
    model = Sequential([       # Sequential 모델을 사용해 층을 순차적으로 쌓습니다.
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),    # 이전 층의 출력을 1차원 배열로 반환
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# 모델을 구축하고 컴파일합니다.
model = build_model((224, 224, 1))  # 입력 형태는 224x224x 크기의 3채널(색상) 이미지
model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=[BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
# compile 메서드를 사용하여 모델을 컴파일한다. 


# 각 에폭의 학습 시간을 기록하기 위한 콜백 클래스입니다.
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
# on_train_begin 메서드는 훈련이 시작될 때 호출되며, 시간 기록을 위한 빈 리스트 times를 초기화합니다.

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()
# on_epoch_begin 메서드는 각 에폭이 시작될 때 호출되며, 에폭의 시작 시간을 기록합니다.

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)
# on_epoch_end 메서드는 각 에폭이 끝날 때 호출되며, 에폭의 총 학습 시간을 times 리스트에 추가합니다.


# TimeHistory 인스턴스를 생성합니다.
time_callback = TimeHistory()

# 모델을 훈련시키면서 시간을 기록합니다.
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[time_callback])

# 모델을 테스트 데이터셋으로 평가하여 성능 지표를 얻습니다.
metrics = model.evaluate(X_test, y_test)

# 학습 및 검증 성능 지표를 그래프로 시각화합니다.
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, metric in enumerate(['accuracy', 'loss', 'precision', 'recall']):
    axes[i].plot(history.history[metric], label=f'Train {metric}')
    axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    axes[i].set_title(f'Epoch vs {metric.title()}')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel(metric.title())
    axes[i].legend()

plt.tight_layout()
plt.show()

# 학습 시간을 그래프로 시각화합니다.
plt.figure(figsize=(8, 5))
plt.plot(time_callback.times)
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.show()

# 최종 모델 성능을 출력합니다.
print(f"Final Model Performance:\n Accuracy: {metrics[1] * 100:.2f}%\n Precision: {metrics[2]:.2f}\n Recall: {metrics[3]:.2f}\n AUC: {metrics[4]:.2f}")
F1_score = 2 * (metrics[2] * metrics[3]) / (metrics[2] + metrics[3])
print(f"F1-score는 {F1_score}이다")









from sklearn.metrics import confusion_matrix
import seaborn as sns

# 테스트 데이터셋에 대한 예측을 수행합니다.
y_pred = model.predict(X_test)
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]  # 임계값을 0.5로 설정하여 이진 분류를 수행합니다.

# Confusion matrix를 생성합니다.
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix를 시각화합니다.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
