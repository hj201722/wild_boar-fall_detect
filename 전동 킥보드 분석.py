import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages
import cv2
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
from utils.plots import plot_one_box

# Firebase 초기화
cred = credentials.Certificate('/Volumes/Hee/kicboard/yolov7/firebase_key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://watchaut-default-rtdb.firebaseio.com/',
    'storageBucket': 'watchaut.appspot.com'
})
bucket = storage.bucket()

# 사용할 장치 설정 (CUDA가 가능하면 CUDA 사용)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 킥보드 탐지 모델 로드
kickboard_model = attempt_load('/Volumes/Hee/kicboard/yolov7/kicboard.pt', map_location=device)
kickboard_model.eval()
kickboard_names = kickboard_model.module.names if hasattr(kickboard_model, 'module') else kickboard_model.names

# 킥보드 번호판 탐지 모델 로드
plate_model = attempt_load('/Volumes/Hee/kicboard/yolov7/licence.pt', map_location=device)
plate_model.eval()
plate_names = plate_model.module.names if hasattr(plate_model, 'module') else plate_model.names

# Google Cloud Vision API 클라이언트 초기화
service_account_file = '/Volumes/Hee/kicboard/yolov7/global-lexicon-406001-bceb8e3785a8.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# 이미지나 비디오 파일 로드 (킥보드 탐지를 위한 이미지 경로)
kickboard_path = '/Volumes/Hee/kicboard/yolov7/1234.jpeg'
kickboard_dataset = LoadImages(kickboard_path, img_size=640)


# 원본 이미지 파일의 경로
original_image_path = kickboard_path  # 이미지 파일 경로를 지정합니다.

# Firebase 스토리지에 원본 이미지 파일 업로드
some_unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
original_image_blob = bucket.blob(f'original_images/kickboard_{some_unique_identifier}.jpg')
original_image_blob.upload_from_filename(original_image_path)
original_image_url = original_image_blob.public_url

# 추론 및 결과 시각화
for path, img, im0s, _ in kickboard_dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 이미지를 float32로 변환
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 추론
    with torch.no_grad():
        kickboard_pred = kickboard_model(img)[0]

    # NMS 적용
    kickboard_pred = non_max_suppression(kickboard_pred, 0.4, 0.5, classes=None, agnostic=False)

    # 탐지된 킥보드 객체 시각화 및 번호판 OCR
    for i, det in enumerate(kickboard_pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # 바운딩 박스 내부의 이미지 영역을 잘라냄
                x_min, y_min, x_max, y_max = [int(coordinate) for coordinate in xyxy]
                crop_img = im0s[y_min:y_max, x_min:x_max]

                #이미지를 메모리상의 바이트 스트림으로 변환
                success, encoded_image = cv2.imencode('.jpg', crop_img)
                content = encoded_image.tobytes()

                #원본 이미지 파일의 경로
                original_image_path = kickboard_path

                # Firebase 스토리지에 이미지 업로드
                some_unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                plate_image_path = f'/Volumes/Hee/kicboard/yolov7/detective_license/kickboard_{some_unique_identifier}.jpg'
                cv2.imwrite(plate_image_path, crop_img)

                blob = bucket.blob(f'plate_images/{some_unique_identifier}.jpg')
                blob.upload_from_filename(plate_image_path)
                plate_image_url = blob.public_url
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


                # # Firebase에 저장
                # data_to_save = {
                #     'Kickboard_image': original_image_url,
                #     'Plate_image': plate_image_url,
                #     'Time' : current_time,
                #     'Location': {
                #         'latitude': 36.7898,
                #         'longitude': 127.0017
                #     }
                # }
                #
                # # Firebase 데이터베이스 참조
                # ref = db.reference('kickboard_results')
                #
                #
                # # 키의 현재 값을 가져와서 1 증가시키는 함수
                # def increment_key_number(current_value):
                #     if current_value is None:
                #         # 만약 현재 값이 없다면, 첫 번째 값을 설정
                #         return 1
                #     else:
                #         # 현재 값에 1을 더하여 반환
                #         return current_value + 1
                #
                #
                # # 트랜잭션을 사용하여 키 값을 증가시키고 데이터 저장
                # key_ref = ref.child('current_key')
                # key_ref.transaction(increment_key_number)
                # current_key = key_ref.get()
                #
                # # 증가된 키를 사용하여 데이터 저장
                # ref.child(f'custom_key_{current_key}').set(data_to_save)

                # 탐지된 텍스트와 함께 바운딩 박스 그림
                label = f'{kickboard_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=(255, 0, 0), line_thickness=3)

        with torch.no_grad():
            plate_pred = plate_model(img)[0]

        #NMS 적용
        plate_pred = non_max_suppression(plate_pred, 0.4, 0.5, classes=None, agnostic=False)

        # 번호판 객체에 대한 후속 작업 수행
        for i, det in enumerate(plate_pred):
            if len(det):
                # 바운딩 박스 좌표를 이미지 크기에 맞게 스케일 조정
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    # 바운딩 박스 내부의 이미지 영역을 잘라냄
                    x_min, y_min, x_max, y_max = [int(coordinate) for coordinate in xyxy]
                    crop_img = im0s[y_min:y_max, x_min:x_max]

                    # 이미지를 메모리상의 바이트 스트림으로 변환
                    success, encoded_image = cv2.imencode('.jpg', crop_img)
                    content = encoded_image.tobytes()

                    # OCR 요청을 실행
                    image = vision.Image(content=content)
                    response = vision_client.text_detection(image=image)
                    texts = response.text_annotations
                    full_text = texts[0].description if texts else ''
                    print("Plate Detection text:", full_text)

                    # Firebase 스토리지에 이미지 업로드
                    some_unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    plate_image_path = f'/Volumes/Hee/kicboard/yolov7/plate_license/licence_{some_unique_identifier}.jpg'
                    cv2.imwrite(plate_image_path, crop_img)

                    blob = bucket.blob(f'plate_images/{some_unique_identifier}.jpg')
                    blob.upload_from_filename(plate_image_path)
                    plate_image_url = blob.public_url
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Firebase 데이터베이스 참조
                    ref = db.reference('kickboard_results')


                    # 현재 저장된 custom_key의 수를 기반으로 새로운 키 생성
                    def get_new_key():
                        result = ref.get()
                        if result:
                            existing_keys = [key for key in result.keys() if key.startswith('kickboard_key_')]
                            return f'kickboard_key_{len(existing_keys) + 1}'
                        else:
                            return 'kickboard_key_1'


                    # 새로운 키 생성
                    new_key = get_new_key()

                    # 데이터 저장할 때의 구조
                    data_to_save = {
                        'Kickboard_image': original_image_url,
                        'Plate_number': full_text,
                        'Time': current_time,
                        'location': {
                            'latitude': 36.7898,
                            'longitude': 127.0017
                        },
                        'plate_image': plate_image_url
                    }

                    # 데이터 저장
                    ref.child(new_key).set(data_to_save)

                    # 탐지된 텍스트와 함께 바운딩 박스 그림
                    label = f'{plate_names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=(0, 0, 255), line_thickness=3)

    # 결과 이미지를 화면에 표시
    cv2.imshow('Result', im0s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()