import argparse
import time
from pathlib import Path
import re
from PIL import Image
import io
import datetime
import random
import string

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import random

import math
import firebase_admin
from firebase_admin import credentials, firestore, storage, db

from models.experimental import attempt_load    # attempt_download는 google_utils.py 에서 거쳐서 옴
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



cred = credentials.Certificate('C:/Users/D4C/Documents/AJS/API_Key/firebase/watchout/watchaut-firebase-adminsdk-m43of-d443b1f435.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://watchaut-default-rtdb.firebaseio.com/',  # 여기에 실제 데이터베이스 URL을 입력
    'storageBucket': 'watchaut.appspot.com'  # 여기에 실제 스토리지 버킷 이름을 입력
})


processed_frames = []

# 높이와 각도 설정
height = 5
angle_degrees = 45
# 각도를 라디안으로 변환
angle_radians = math.radians(angle_degrees)
# 하단의 길이 계산
base_length = height * math.tan(angle_radians)
# 면적 계산
area = 0.5 * base_length * height
area = round(area, 3)   # 12.5


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            
            # 이미지에 텍스트 그리기
            # cv2.putText(im0, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if cls == 0:  # 사람 클래스인 경우에만 처리
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # print('stingt: ', s)
            num_of_p = re.search(r'(\d+)\sperson', s).group(1)
            # processed_frames.append(im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    
        # 이미지 url
        image_url = upload_image_to_cloud(im0, p.name)
        
        num_of_p_float = float(num_of_p)  # 모든 요소를 float으로 변환
        # 데이터를 실시간 데이터베이스에 저장
        save_data_to_realtime_database("송악면", area, num_of_p_float, "와차웃 기차역", image_url)
        

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



# 이미지를 클라우드에 업로드하는 함수
def upload_image_to_cloud(image_data, image_name):
    # 이미지를 저장할 버킷과 경로 설정
    bucket = storage.bucket()

    # 이미지 데이터가 리스트인 경우와 단일 이미지 데이터인 경우를 처리
    if isinstance(image_data, list):
        # 이미지 데이터가 리스트인 경우
        public_urls = []
        for index, image in enumerate(image_data):
            # 고유한 파일 이름 생성
            unique_filename = f"{image_name}_{index}.jpg"

            # OpenCV 이미지를 JPEG 형식으로 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            byte_stream = io.BytesIO(buffer)

            # 고유한 이름으로 이미지를 업로드하고 URL을 가져옴
            single_blob = bucket.blob(f'density_img/{unique_filename}')
            single_blob.upload_from_file(byte_stream, content_type='image/jpeg')
            single_blob.make_public()
            public_urls.append(single_blob.public_url)  # URL을 배열에 추가

        return public_urls

    else:   # 단일 이미지 데이터인 경우
        unique_filename = generate_unique_filename(image_name)

        # OpenCV 이미지를 JPEG 형식으로 인코딩
        _, buffer = cv2.imencode('.jpg', image_data)
        byte_stream = io.BytesIO(buffer)

        # 이미지를 업로드하고 URL을 가져옴
        blob = bucket.blob(f'density_img/{unique_filename}')
        blob.upload_from_file(byte_stream, content_type='image/jpeg')
        blob.make_public()

        return blob.public_url



def generate_unique_filename(base_name):
    # 현재 시간과 랜덤 문자열을 조합하여 유니크한 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{base_name}_{timestamp}_{random_str}.jpg"



def save_data_to_realtime_database(town, area, num_of_ps, region_name, url):
    # 현재 시간을 가져오기
    current_time = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")

    # num_of_ps와 url이 리스트인 경우, 첫 번째 요소 사용
    if isinstance(num_of_ps, list):
        num_of_ps = num_of_ps[0]

    if isinstance(url, list):
        url = url[0]

    # 밀집도와 점유율 계산
    possession_rate = round(num_of_ps * 0.5 / area * 100, 4)  # 성인이 평균적으로 0.5 제곱미터를 차지한다고 가정
    density_rate = round(num_of_ps / area, 4)

    # Firebase Realtime Database에 데이터 저장
    region_ref = db.reference('Density').child(town).child(region_name).child(current_time)
    data = {
        'url': url,
        'num_of_p': num_of_ps,
        'density_rate': density_rate,
        'possession_rate': possession_rate
    }
    region_ref.set(data)
    
    
    # main page 이미지 실시간으로 하기 위해 db의 저장 경로를 임시로 잡아주는 코드
    # region_ref = db.reference('Main_image').child(current_time)
    # data = {
    #     'url': url
    # }
    # region_ref.set(data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-e6e.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='../test_img/density_cctv.mp4', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='../test_img/seq_000104.jpg', help='source')
    # parser.add_argument('--source', type=str, default='../test_img/yolo_test4.mp4', help='source')
    # parser.add_argument('--source', type=str, default='../test_img/wildboar.mp4', help='source')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    
