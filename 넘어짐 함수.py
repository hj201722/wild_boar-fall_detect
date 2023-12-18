import firebase_admin
from firebase_admin import credentials, db, storage
import datetime
import cv2
import logging
import io

# Firebase 초기화
def initialize_firebase(key_path):
    try:
        global db, bucket
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://watchaut-default-rtdb.firebaseio.com/',
                'storageBucket': 'watchaut.appspot.com'
            })
        bucket = storage.bucket()
    except Exception as e:
        logging.error(f"Error initializing Firebase: {e}")
        exit(1)

# 카메라 정보 가져오기
def get_camera_info(camera_id):
    try:
        ref = db.reference(f'Camera/{camera_id}')
        camera_info = ref.get()
        if camera_info:
            latitude = camera_info.get('Latitude')
            longitude = camera_info.get('Longitude')
            status = camera_info.get('Status')
            return latitude, longitude, status
        else:
            return None, None, None
    except Exception as e:
        logging.error(f"Error in get_camera_info: {e}")
        return None, None, None
    
    
def get_latest_event_counter():
    try:
        ref = db.reference('FellPerson')
        all_events = ref.get()
        if all_events:
            highest_event_id = max(int(event_id.replace('fellperson',''))for event_id in all_events.keys())
            return highest_event_id + 1
        else:
            return 1
    except Exception as e:
        logging.error(f"Error in get_latest_event_counrter: {e}")
        return 1


def save_image_to_storage(frame, file_name):
    try:
        _, buffer = cv2.imencode('.jpg',frame)
        byte_stream = io.BytesIO(buffer)
        
        folder_name = "FALL"
        blob = storage.bucket().blob(f'{folder_name}/{file_name}')
        blob.upload_from_file(byte_stream, content_type='image/jpeg')
        
        expiration = datetime.timedelta(hours=24)
        image_url = blob.generate_signed_url(expiration = expiration)
        return image_url
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return None

# 실시간 데이터베이스에 로깅
def log_event_to_realtime_db(latitude, longitude, status, image_url, event_counter):
    try:
        now = datetime.datetime.now()
        event_id = f"fellperson{str(event_counter).zfill(3)}"
        ref = db.reference(f'FellPerson/{event_id}')
        ref.set({
            'Latitude': latitude,
            'Longitude': longitude,
            'Status': status,
            'Time': now.strftime("%Y-%m-%d %H:%M:%S"),
            'ImageURL': image_url
        })
    except Exception as e:
        logging.error(f"Error in log_event_to_realtime_db: {e}")
