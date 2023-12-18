import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# 각 비디오에서 프레임 추출을 위한 함수
def extract_frames(video_path, frame_count):
    frames = [] 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수를 얻습니다.

    print(f'Processing video: {video_path}')
    
    # 추출할 프레임의 간격을 계산합니다.
    frame_gap = total_frames / float(frame_count)
    
    current_frame = 0
    for f in range(frame_count):
        # 읽을 프레임의 위치를 설정합니다.
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(current_frame))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_AREA)
            frame = frame / 255.0  # 정규화
            frame = frame.astype(np.float32)  # 데이터 타입을 float32로 변경
            frames.append(frame)
            print(f"Extracted frame {len(frames)} / {frame_count}")
        else:
            print("Error reading frame.")
            break
        current_frame += frame_gap

    cap.release()
    return np.array(frames)
    

# 새로운 비디오 데이터로부터 예측을 수행하고 폭력이 감지된 프레임을 저장하는 함수
def predict_and_save_frames(video_path, model, frame_count, img_height, img_width, num_classes, threshold=0.9):
    frames = extract_frames(video_path, frame_count)
    if len(frames) == frame_count:
        frames_for_prediction = frames[np.newaxis, ...]  # 모델이 받아들일 수 있는 차원으로 조정합니다.
        predictions = model.predict(frames_for_prediction)
        predicted_classes = np.argmax(predictions, axis=-1)
        
        for i, (frame, prediction) in enumerate(zip(frames, predictions)):
            if prediction[0] > threshold and predicted_classes[i] == 0:  # 폭력으로 예측된 경우
                violence_probability = prediction[0]
                frame_to_save = (frame * 255).astype('uint8')  # 정규화된 프레임을 원래의 0-255 범위로 되돌립니다.
                cv2.imwrite(f'violence_frame_{i+1}_{violence_probability:.2f}.jpg', frame_to_save)
                print(f'Saved violence frame {i+1} with probability {violence_probability:.2f}')
        
        predicted_class_name = 'Violence' if predicted_classes[0] == 0 else 'Nonviolence'
        return predicted_class_name, predictions[0]
    else:
        print("Couldn't predict video due to insufficient frames.")
        return None, None

# 데이터셋 준비
def prepare_data(folder_name, frame_count):
    X, y = [], []
    classes = ['Violence', 'Nonviolence']
    for class_index, class_name in enumerate(classes):
        class_folder = os.path.join(folder_name, class_name)
        files = os.listdir(class_folder)
        print(f'Processing class: {class_name}')
        for file_index, file in enumerate(files):  # 파일 인덱스를 추가하여 로깅합니다.
            video_path = os.path.join(class_folder, file)
            print(f'Processing video {file_index + 1}/{len(files)}: {file}')  # 진행 상황을 출력합니다.
            frames = extract_frames(video_path, frame_count)
            if len(frames) == frame_count:
                X.append(frames)
                y.append(class_index)
            else:
                print(f'Video {video_path} was skipped due to insufficient frames.')  # 프레임 수가 부족한 비디오를 건너뛰고 로깅합니다.
    return np.array(X), np.array(y)

# 모델 구축
def build_lstm_cnn_model(frame_count, img_height, img_width, num_classes):
    model = Sequential()
    # TimeDistributed CNN 레이어
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(frame_count, img_height, img_width, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    # LSTM 레이어
    model.add(LSTM(50))
    # 분류 레이어
    model.add(Dense(num_classes, activation='softmax'))
    # 컴파일
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 하이퍼파라미터 설정
frame_count = 500 # 추출할 프레임 수
img_height, img_width = 416,416
num_classes = 2

tf.debugging.set_log_device_placement(True)

# 데이터 준비
X, y = prepare_data('dataset', frame_count)
y = to_categorical(y, num_classes) # 원핫 인코딩

# 훈련 및 검증 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구축
model = build_lstm_cnn_model(frame_count, img_height, img_width, num_classes)

# 모델 훈련
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=5)

# 모델 평가
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)
print(classification_report(y_true_classes, y_pred_classes))

# 테스트 데이터셋 준비
#X_test, y_test = prepare_data('Test', frame_count)
#y_test = to_categorical(y_test, num_classes)

# 테스트 데이터셋으로 모델 평가
#test_loss, test_accuracy = model.evaluate(X_test, y_test)
#print(f"Test accuracy: {test_accuracy}")

# 모델 저장
model.save('violence_detection_model.h5')

video_path = 'fight1.mp4'

predicted_class_name, predictions = predict_and_save_frames(video_path, model, frame_count, img_height, img_width, num_classes)

# 예측 결과 출력
if predicted_class_name is not None:
    print(f"The video is predicted to be a '{predicted_class_name}' video.")
    print(f"Prediction probabilities: {predictions}")
else:
    print("Failed to make a prediction.")