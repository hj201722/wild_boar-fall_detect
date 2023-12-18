from flask import Flask, request, render_template
import firebase_admin
from firebase_admin import credentials, db, storage
import subprocess
import os
import datetime

app = Flask(__name__, static_folder='templates/assets')


def initialize_firebase():
    # Firebase Admin SDK 초기화
    cred = credentials.Certificate('watchaut-firebase-adminsdk-m43of-7aabebaefe.json')
    firebase_app = firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://watchaut-default-rtdb.firebaseio.com/',
        'storageBucket': 'watchaut.appspot.com'
    })
    return firebase_app


def get_next_file_name(bucket, directory):
    # 해당 디렉토리에 있는 파일 목록 가져오기
    blobs = bucket.list_blobs(prefix=directory)
    file_count = 0
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            file_count += 1

    # 새 파일 이름 생성 (예: 001.jpg, 002.jpg)
    next_file_number = file_count + 1
    return f"{str(next_file_number).zfill(3)}.jpg"


# 파이어베이스 스토리지에 이미지를 저장하고 url을 리턴
def upload_file_to_firebase(file_path, target_objects):
    bucket = storage.bucket()
    directory = f"MainObject/{target_objects}"

    # 새 파일 이름 결정
    new_file_name = get_next_file_name(bucket, directory)
    blob = bucket.blob(f"{directory}/{new_file_name}")
    blob.upload_from_filename(file_path)
    
    # Firebase 스토리지에 접근하기 위한 URL 생성
    # 이 URL은 권한 토큰을 포함하여 외부에서 접근이 가능하게 합니다.
    image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media&token={blob.generate_signed_url(expiration=datetime.timedelta(seconds=300), version='v4')}"

    return image_url


@app.route('/', methods=['GET', 'POST'])
def process_frame():
    if request.method == 'POST':
        try:
            # 파일 저장 경로 확인 및 생성
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)

            # Firebase에서 target_objects 값 가져오기
            ref = db.reference('Main/object')
            target_objects = ref.get()

            frame = request.files['frame']
            frame_path = os.path.join(upload_folder, 'frame.jpg')
            frame.save(frame_path)

            output_path = os.path.join(upload_folder, 'output')

            os.environ['CUDA_VISIBLE_DEVICES'] = "0"

            current_path = os.path.dirname(os.path.abspath(__file__))
            os.environ['PYTHONPATH'] = current_path + os.pathsep + os.environ.get('PYTHONPATH', '')

            command = f'python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i {frame_path} -o {output_path} -t "{target_objects}"'
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 이미지 처리가 완료된 후, Firebase Storage에 업로드
            processed_image_path = os.path.join(output_path, 'pred.jpg')
            image_url = upload_file_to_firebase(processed_image_path, target_objects)   # 스토리지에 저장된 이미지의 url
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            link_ref =  db.reference('Main_image').child(current_time)
            
            # URL을 Realtime Database에 저장
            link_ref.set({'url': image_url})
            

            return result.stdout
        except subprocess.CalledProcessError as e:
            return e.stderr, 500

    return render_template('index2.html')




if __name__ == '__main__':
    initialize_firebase()
    app.run(host='0.0.0.0', port=5001, debug=True)