import cv2
import numpy as np
import onnxruntime
import time
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# MP4 파일에서 해상도 읽어오기
def get_video_resolution(video_path):
    """
    동영상 파일에서 해상도를 읽어옵니다.

    Parameters:
    - video_path (str): 동영상 파일 경로.

    Returns:
    - tuple: (width, height) 동영상의 해상도 (너비, 높이).

    Raises:
    - ValueError: 파일 경로가 유효하지 않거나 동영상을 열 수 없는 경우.
    """
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # 해상도 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 객체 해제
    cap.release()

    # 해상도 반환
    return width, height


def preprocess(image, input_shape):
    """
    이미지를 YOLO 입력 형식에 맞게 전처리합니다.
    - 이미지를 지정된 크기에 맞춰 리사이즈 및 패딩 처리.
    - 픽셀 값을 [0, 1] 범위로 정규화.
    - 데이터 차원을 YOLO 입력 형식인 CHW (채널, 높이, 너비)로 변환.

    Parameters:
    - image (np.ndarray): 원본 이미지 (HWC 형식, dtype=np.uint8).
    - input_shape (tuple): YOLO 모델 입력 크기 (width, height).

    Returns:
    - np.ndarray: YOLO 입력 형식으로 변환된 이미지 (NCHW 형식, dtype=np.float32).
    - float: 리사이즈 시 사용된 스케일 값.
    - tuple: 리사이즈된 이미지의 크기 (width, height).
    """
    # YOLO 입력 크기
    input_w, input_h = input_shape

    # 원본 이미지 크기
    original_h, original_w = image.shape[:2]

    # 리사이즈를 위한 스케일 계산
    scale = min(input_w / original_w, input_h / original_h)
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)

    # 이미지 리사이즈
    resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # 패딩된 YOLO 입력 크기 이미지 생성 (배경색은 중립 회색 [128, 128, 128])
    padded_image = np.full((input_h, input_w, 3), 128, dtype=np.float32)

    # 리사이즈된 이미지를 패딩된 이미지 중앙에 복사
    start_h = (input_h - scaled_h) // 2
    start_w = (input_w - scaled_w) // 2
    padded_image[start_h:start_h + scaled_h, start_w:start_w + scaled_w, :] = resized_image

    # 픽셀 값을 [0, 1]로 정규화
    padded_image = padded_image / 255.0

    # CHW 형식으로 변환
    padded_image = np.transpose(padded_image, (2, 0, 1))

    # 배치 차원을 추가하여 NCHW 형식으로 변환 후 반환
    return np.expand_dims(padded_image, axis=0).astype(np.float32), scale, (scaled_w, scaled_h)


def non_max_suppression(prediction, scale_tuple, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                        multi_label=False, labels=()):
    """
    Non-Maximum Suppression (NMS)을 적용하여 중복된 탐지 결과를 제거합니다.

    Parameters:
    - prediction (Tensor | np.ndarray): 탐지 결과를 포함한 텐서 또는 배열.
      각 행은 [cx, cy, w, h, objectness_score, class_scores...]로 구성.
    - scale_tuple (tuple): 스케일 관련 정보 (scale_factor, scaled_h, scaled_w).
    - conf_thres (float): 신뢰도 임계값 (confidence threshold).
    - iou_thres (float): IOU 임계값 (Intersection Over Union threshold).
    - classes (list, optional): 유지할 클래스 ID 목록. 기본값은 None (모든 클래스 유지).
    - agnostic (bool, optional): True일 경우 클래스 무관하게 NMS 적용.
    - multi_label (bool, optional): True일 경우 하나의 박스에 여러 클래스 라벨 허용.
    - labels (list, optional): 미리 정의된 라벨(자동 라벨링용).

    Returns:
    - box_list (list): 필터링된 바운딩 박스 리스트. 각 항목은 Tensor 형식.
    - scores_list (list): 필터링된 점수와 클래스 리스트. 각 항목은 Tensor 형식.
    """
    # 입력이 리스트나 튜플 형태일 경우 첫 번째 요소 선택
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    # 입력이 NumPy 배열일 경우 PyTorch Tensor로 변환
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).to('cuda' if torch.cuda.is_available() else 'cpu')

    box_list, scores_list = [], []

    # 클래스 개수와 신뢰도 임계값 적용
    nc = prediction.shape[2] - 5  # 클래스 개수
    xc = prediction[..., 4] > conf_thres  # 신뢰도 필터링

    # 기본 설정값
    min_wh, max_wh = 2, 4096  # 최소 및 최대 박스 크기
    max_det = 300  # 최대 탐지 개수
    max_nms = 30000  # NMS에 사용할 최대 박스 개수
    time_limit = 10.0  # NMS 시간 제한
    redundant = True  # 중복 제거 활성화
    multi_label &= nc > 1  # 다중 라벨 허용 조건
    merge = False  # 병합 NMS 비활성화

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    # 각 이미지에 대해 처리
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # 신뢰도 필터링

        # 라벨 추가 (자동 라벨링)
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 박스 좌표
            v[:, 4] = 1.0  # 신뢰도
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 클래스
            x = torch.cat((x, v), 0)

        # 필터링 결과가 없는 경우
        if not x.shape[0]:
            box_list.append([])
            scores_list.append([])
            continue

        # 신뢰도 계산
        if nc == 1:
            x[:, 5:] = x[:, 4:5]
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = objectness_score * class_score

        # 바운딩 박스 좌표 변환
        box = xywh2xyxy(x[:, :4], scale_tuple)

        # 클래스별 신뢰도 필터링
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 특정 클래스 필터링
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 박스 수 확인
        n = x.shape[0]
        if not n:
            box_list.append([])
            scores_list.append([])
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS 적용
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # 병합 NMS
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

        # 결과 저장
        box_list.append(output[xi][:, :4].cpu())
        scores_list.append(output[xi][:, 4:].cpu())

    return box_list, scores_list


def xywh2xyxy(x, scale_tuple):
    """
    중심 좌표와 크기로 표현된 바운딩 박스 (x_center, y_center, width, height)를
    좌상단과 우하단 좌표로 변환 (x_min, y_min, x_max, y_max)합니다.

    Parameters:
    - x (array or tensor): 변환할 바운딩 박스. 각 행은 [x_center, y_center, width, height]로 표현.
    - scale_tuple (tuple): 스케일 정보 (scale_factor, scaled_h, scaled_w).

    Returns:
    - y (array or tensor): 변환된 바운딩 박스. 각 행은 [x_min, y_min, x_max, y_max]로 표현.
    """
    # 스케일 정보를 해석
    scale_factor, scaled_h, scaled_w = scale_tuple

    # 입력이 Tensor인지 확인 후, 해당 타입에 맞는 복사본 생성
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # 이미지가 정사각형이 아닌 경우, 크기를 조정
    if scaled_w != scaled_h:
        scale_ratio = scaled_w / scaled_h if scaled_w > scaled_h else scaled_h / scaled_w
        offset = (abs(scaled_w - scaled_h) // 2)  # 비율 차이 계산

        if scaled_w > scaled_h:
            x[..., 0] = (x[..., 0] - offset) * scale_ratio  # x_center 조정
            x[..., 2] *= scale_ratio  # width 조정
        else:
            x[..., 1] = (x[..., 1] - offset) * scale_ratio  # y_center 조정
            x[..., 3] *= scale_ratio  # height 조정

    # (cx, cy, w, h)를 (x_min, y_min, x_max, y_max)로 변환
    y[..., 0] = torch.clamp(x[..., 0] - x[..., 2] / 2, min=0)  # top-left x
    y[..., 1] = torch.clamp(x[..., 1] - x[..., 3] / 2, min=0)  # top-left y
    y[..., 2] = torch.clamp(x[..., 0] + x[..., 2] / 2, min=0)  # bottom-right x
    y[..., 3] = torch.clamp(x[..., 1] + x[..., 3] / 2, min=0)  # bottom-right y

    return y



COCO_2017_TO_2014_TRANSLATION = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
    11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20,
    19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
    27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39,
    35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48,
    43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
    51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64,
    59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76,
    67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
    75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}

def translate_coco_2017_to_2014(nmsed_classes):
    """
    COCO 2017 클래스 ID를 COCO 2014 클래스 ID로 변환합니다.

    Parameters:
    - nmsed_classes (numpy.ndarray): COCO 2017 클래스 ID를 담은 numpy 배열.

    Returns:
    - numpy.ndarray: COCO 2014 클래스 ID로 변환된 numpy 배열.
    """
    # COCO_2017_TO_2014_TRANSLATION 딕셔너리를 사용해 변환
    return np.vectorize(COCO_2017_TO_2014_TRANSLATION.get)(nmsed_classes).astype(np.int32)


def get_label(class_id):
    """
    COCO 데이터셋의 클래스 ID에 해당하는 레이블을 반환합니다.

    Parameters:
    - class_id (int): COCO 데이터셋의 클래스 ID.

    Returns:
    - label (str): 클래스 ID에 해당하는 레이블 이름.

    Notes:
    - 'coco2017.txt' 파일에는 COCO 2017 데이터셋 클래스 ID와 레이블 이름의 매핑이 저장되어 있어야 합니다.
    - 파일 형식은 Python의 딕셔너리 형태여야 하며, 다음과 같은 형식을 가져야 합니다:
        {0: "person", 1: "bicycle", 2: "car", ...}
    """
    # COCO 2017 클래스와 레이블 매핑 파일 경로
    coco_labels = 'coco2017.txt'

    # 매핑 파일 열기 및 클래스 ID로 레이블 검색
    with open(coco_labels, 'r') as f:
        labels = eval(f.read())  # 파일 내용을 Python 딕셔너리로 변환
        return labels[class_id]


def post_process(boxes, scores, classes, width, height):
    """
    바운딩 박스 탐지 결과를 처리하여 시각화와 포즈 필터링에 적합한 데이터를 생성합니다.

    Parameters:
    - boxes (list of list): 정규화된 좌표 형식의 바운딩 박스 리스트 [(x_min, y_min, x_max, y_max), ...].
    - scores (list): 각 바운딩 박스의 신뢰도 점수 리스트.
    - classes (list): 각 바운딩 박스의 클래스 ID 리스트.
    - width (int): 원본 이미지의 너비.
    - height (int): 원본 이미지의 높이.

    Returns:
    - draw_box (list of dict): 시각화를 위한 바운딩 박스 세부 정보 리스트.
      각 항목은 딕셔너리 형식으로, 다음 키를 포함합니다:
        - "label": 클래스 이름과 신뢰도 점수를 포함한 레이블.
        - "confidence": 신뢰도 점수.
        - "color_idx": 클래스 ID.
        - "pt1": 바운딩 박스의 좌측 상단 좌표 (픽셀 단위).
        - "pt2": 바운딩 박스의 우측 하단 좌표 (픽셀 단위).
        - "label_pt": 레이블 텍스트 좌표 (픽셀 단위).
    - pose_box (list of dict): "person" 레이블이 있는 바운딩 박스의 필터링된 리스트.
    """
    draw_box = []  # 시각화용 바운딩 박스 정보
    pose_box = []  # "person" 바운딩 박스 정보

    # 각 바운딩 박스 처리
    for idx in range(len(boxes)):
        # 바운딩 박스 정보 생성
        tmp_draw_box = {}
        label = get_label(classes[idx]) + f": {scores[idx] * 100:.1f}%"  # 클래스 이름 및 신뢰도
        tmp_draw_box["label"] = label
        tmp_draw_box["confidence"] = scores[idx]
        tmp_draw_box["color_idx"] = classes[idx]

        # 정규화된 좌표를 이미지 크기에 맞게 변환
        x_min, y_min, x_max, y_max = boxes[idx]
        tmp_draw_box["pt1"] = (int(x_min * width), int(y_min * height))  # 좌측 상단 좌표
        tmp_draw_box["pt2"] = (int(x_max * width), int(y_max * height))  # 우측 하단 좌표
        tmp_draw_box["label_pt"] = (int(x_min * width), int(y_min * height) - 10)  # 레이블 텍스트 좌표

        # 바운딩 박스 정보 추가
        draw_box.append(tmp_draw_box)

        # "person" 클래스인 경우 pose_box에 추가
        if "person" in label:
            pose_box.append(tmp_draw_box)

    return draw_box, pose_box


def draw_boxes_n_skeleton(image, draw_box_list, pose_box_list, outs):
    """
    이미지에 탐지된 바운딩 박스, 레이블, 그리고 포즈 추정 결과를 그립니다.

    Parameters:
    - image (np.ndarray): 입력 이미지.
    - draw_box_list (list): 탐지된 객체의 바운딩 박스 정보 리스트.
    - pose_box_list (list): 탐지된 "person" 포즈 정보 리스트.
    - outs (list): 포즈 추정 결과 (키포인트 및 스켈레톤 정보 리스트).

    Returns:
    - image (np.ndarray): 바운딩 박스, 레이블, 그리고 포즈가 그려진 이미지.
    """
    # 검출된 객체가 없으면 아무 작업도 수행하지 않음
    if not draw_box_list or draw_box_list == [[]]:
        return image
    if not pose_box_list or pose_box_list == [[]]:
        return image

    # 탐지된 객체의 바운딩 박스를 그리기
    for box_group in draw_box_list:
        for det in box_group:
            x1, y1 = det["pt1"]
            x2, y2 = det["pt2"]
            label = det.get("label", "")

            # 바운딩 박스 그리기 (녹색)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 레이블 텍스트 그리기
            if label:
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # "person" 바운딩 박스를 그리기
    for box_group in pose_box_list:
        for det in box_group:
            x1, y1 = det["pt1"]
            x2, y2 = det["pt2"]
            label = det.get("label", "")

            # 바운딩 박스 그리기 (파란색)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 레이블 텍스트 그리기
            if label:
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 포즈 추정 결과 (키포인트 및 스켈레톤) 그리기
    for pts in outs:
        image = draw_points_and_skeleton(
            image,
            pts,
            joints_dict()["coco"]["skeleton"],  # COCO 스켈레톤 구조 사용
            person_index=0,
            points_color_palette="gist_rainbow",
            skeleton_color_palette="jet",
            points_palette_samples=10,
        )

    return image


def transform(image):
    """
    이미지를 추론에 사용할 수 있도록 정규화 및 형식 변환을 수행합니다.

    Parameters:
    - image (np.ndarray): BGR 형식의 입력 이미지 (H, W, C).

    Returns:
    - np.ndarray: 변환된 이미지로, 형식은 [C, H, W].
    """
    # 1. 이미지를 float32로 변환하고 [0, 1] 범위로 정규화
    image = image.astype(np.float32) / 255.0

    # 2. 평균(mean) 및 표준편차(std)로 정규화 (ImageNet 사전 학습 모델의 기준값 사용)
    mean = np.array([0.485, 0.456, 0.406])  # RGB 채널 평균
    std = np.array([0.229, 0.224, 0.225])   # RGB 채널 표준편차
    image = (image - mean) / std

    # 3. HWC 형식의 이미지를 CHW 형식으로 변환
    return np.transpose(image, (2, 0, 1))


def crop_image(x1, x2, y1, y2, img, pose_resolution):
    """
    바운딩 박스를 기준으로 이미지를 크롭하고, 포즈 추정을 위한 해상도 비율에 맞게 조정합니다.

    Parameters:
    - x1, x2, y1, y2 (int): 바운딩 박스 좌표 (왼쪽 위 x, 오른쪽 아래 x, 왼쪽 위 y, 오른쪽 아래 y).
    - img (np.ndarray): 원본 이미지.
    - pose_resolution (tuple): 포즈 추정을 위한 목표 해상도 (높이, 너비).

    Returns:
    - image_crop (np.ndarray): 크롭된 이미지.
    - pad_tuple (tuple): 이미지에 적용된 패딩 정보.
    - x1_new, x2_new, y1_new, y2_new (int): 조정된 바운딩 박스 좌표.
    """
    # 크롭 비율 보정을 위한 보정 계수 계산
    correction_factor = pose_resolution[0] / pose_resolution[1] * (x2 - x1) / (y2 - y1)

    # 세로를 확장해야 하는 경우
    if correction_factor > 1:
        center = y1 + (y2 - y1) // 2
        length = int(round((y2 - y1) * correction_factor))
        x1_new, x2_new = x1, x2
        y1_new, y2_new = int(center - length // 2), int(center + length // 2)
        pad = (abs(y1_new - y1), abs(y2_new - y2))
        pad_tuple = (pad, (0, 0), (0, 0))

    # 가로를 확장해야 하는 경우
    elif correction_factor < 1:
        center = x1 + (x2 - x1) // 2
        length = int(round((x2 - x1) * (1 / correction_factor)))
        x1_new, x2_new = int(center - length // 2), int(center + length // 2)
        y1_new, y2_new = y1, y2
        pad = (abs(x1_new - x1), abs(x2_new - x2))
        pad_tuple = ((0, 0), pad, (0, 0))

    # 보정이 필요 없는 경우
    else:
        x1_new, x2_new = x1, x2
        y1_new, y2_new = y1, y2
        pad_tuple = None

    # 이미지 크롭
    image_crop = img[y1:y2, x1:x2, ::]
    return image_crop, pad_tuple, x1_new, x2_new, y1_new, y2_new


def joints_post_process(out, box, resized_pose_resolution):
    """
    포즈 추정 모델의 출력을 원본 이미지 공간으로 변환합니다.

    Parameters:
    - out (np.ndarray): 각 조인트에 대한 히트맵 출력 (n_joints, height, width).
    - box (dict): 바운딩 박스 정보가 포함된 딕셔너리로, "pt1" (좌상단)과 "pt2" (우하단) 좌표를 포함.
    - resized_pose_resolution (tuple): 포즈 추정을 위한 해상도 (height, width).

    Returns:
    - pts (np.ndarray): (n_joints, 3) 형식의 배열. 각 행은 [x_position, y_position, confidence_score]를 나타냅니다.
    """
    # 출력 조인트 수에 맞는 빈 배열 생성
    n_joints = out.shape[0]
    pts = np.empty((n_joints, 3), dtype=np.float32)

    # 바운딩 박스 좌표 추출
    x_min, y_min = box["pt1"]
    x_max, y_max = box["pt2"]

    # 각 조인트의 히트맵에서 가장 큰 값의 위치와 해당 값을 계산
    for j in range(n_joints):
        joint_heatmap = out[j]
        joint_max = np.argmax(joint_heatmap)  # 가장 큰 값의 인덱스
        pt = (joint_max // resized_pose_resolution[1],  # y 좌표 (행)
              joint_max % resized_pose_resolution[1])   # x 좌표 (열)

        # 원본 이미지 좌표로 변환
        pts[j, 0] = pt[1] * (x_max - x_min) / resized_pose_resolution[1] + x_min  # x 좌표
        pts[j, 1] = pt[0] * (y_max - y_min) / resized_pose_resolution[0] + y_min  # y 좌표
        pts[j, 2] = joint_heatmap[pt]  # 신뢰도 (confidence score)

    return pts


def draw_points_and_skeleton(image, keypoints, skeleton, person_index=0,
                             points_color_palette='gist_rainbow',
                             skeleton_color_palette='jet',
                             points_palette_samples=10):
    """
    이미지에 키포인트와 스켈레톤을 그립니다.

    Parameters:
    - image: OpenCV 이미지.
    - keypoints: 키포인트 배열 (n_joints, 3) 형식, [x, y, confidence].
    - skeleton: 스켈레톤을 나타내는 연결 리스트, [(joint1, joint2), ...].
    - person_index: 사람 인덱스 (기본값 0).
    - points_color_palette: 키포인트 색상 팔레트 (matplotlib 팔레트 이름).
    - skeleton_color_palette: 스켈레톤 색상 팔레트 (matplotlib 팔레트 이름).
    - points_palette_samples: 팔레트에서 샘플링할 색상 수.

    Returns:
    - image: 키포인트와 스켈레톤이 그려진 이미지.
    """
    # 키포인트와 스켈레톤 색상 팔레트 설정
    points_cmap = plt.colormaps.get_cmap(points_color_palette)
    skeleton_cmap = plt.colormaps.get_cmap(skeleton_color_palette)
    point_colors = [points_cmap(i / points_palette_samples)[:3] for i in range(points_palette_samples)]
    skeleton_colors = [skeleton_cmap(i / len(skeleton))[:3] for i in range(len(skeleton))]

    # 키포인트를 그리기
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  # 신뢰도 50% 이상일 때만 그리기
            color = (np.array(point_colors[idx % points_palette_samples]) * 255).astype(int).tolist()
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            cv2.putText(image, f"{idx}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 스켈레톤을 그리기
    for idx, (joint1, joint2) in enumerate(skeleton):
        # 범위 확인 및 신뢰도 확인
        if joint1 < len(keypoints) and joint2 < len(keypoints):
            if keypoints[joint1][2] > 0.5 and keypoints[joint2][2] > 0.5:  # 두 점 q모두 신뢰도 50% 이상
                x1, y1, _ = keypoints[joint1]
                x2, y2, _ = keypoints[joint2]
                color = (np.array(skeleton_colors[idx]) * 255).astype(int).tolist()
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return image

def joints_dict():
    """
    포즈 추정을 위한 데이터셋의 키포인트 정의 및 스켈레톤 연결 정보를 제공합니다.

    Returns:
    - joints: 키포인트 정의와 스켈레톤 연결 정보를 담고 있는 딕셔너리.
      - "coco": COCO 데이터셋의 17개 키포인트 및 스켈레톤 연결 정보.
      - "mpii": MPII 데이터셋의 16개 키포인트 및 스켈레톤 연결 정보.
    """
    joints = {
        "coco": {
            "keypoints": {  # COCO 데이터셋 키포인트 정의 (인덱스: 부위 이름)
                0: "nose",           # 코
                1: "left_eye",       # 왼쪽 눈
                2: "right_eye",      # 오른쪽 눈
                3: "left_ear",       # 왼쪽 귀
                4: "right_ear",      # 오른쪽 귀
                5: "left_shoulder",  # 왼쪽 어깨
                6: "right_shoulder", # 오른쪽 어깨
                7: "left_elbow",     # 왼쪽 팔꿈치
                8: "right_elbow",    # 오른쪽 팔꿈치
                9: "left_wrist",     # 왼쪽 손목
                10: "right_wrist",   # 오른쪽 손목
                11: "left_hip",      # 왼쪽 엉덩이
                12: "right_hip",     # 오른쪽 엉덩이
                13: "left_knee",     # 왼쪽 무릎
                14: "right_knee",    # 오른쪽 무릎
                15: "left_ankle",    # 왼쪽 발목
                16: "right_ankle"    # 오른쪽 발목
            },
            "skeleton": [  # COCO 데이터셋 스켈레톤 연결 정의 (키포인트 간 연결)
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                [1, 3], [2, 4], [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {  # MPII 데이터셋 키포인트 정의 (인덱스: 부위 이름)
                0: "right_ankle",   # 오른쪽 발목
                1: "right_knee",    # 오른쪽 무릎
                2: "right_hip",     # 오른쪽 엉덩이
                3: "left_hip",      # 왼쪽 엉덩이
                4: "left_knee",     # 왼쪽 무릎
                5: "left_ankle",    # 왼쪽 발목
                6: "pelvis",        # 골반
                7: "thorax",        # 흉곽
                8: "upper_neck",    # 목
                9: "head top",      # 머리 꼭대기
                10: "right_wrist",  # 오른쪽 손목
                11: "right_elbow",  # 오른쪽 팔꿈치
                12: "right_shoulder", # 오른쪽 어깨
                13: "left_shoulder",  # 왼쪽 어깨
                14: "left_elbow",   # 왼쪽 팔꿈치
                15: "left_wrist"    # 왼쪽 손목
            },
            "skeleton": [  # MPII 데이터셋 스켈레톤 연결 정의
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2],
                [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10]
            ]
        }
    }
    return joints


def process_video(video_path, YOLO_model_path, YOLO_input_shape, HRNet_model_path, HRNet_input_shape):
    """
    동영상을 처리하며 YOLO 모델을 사용해 객체를 탐지하고 HRNet 모델로 포즈를 추정하여 결과를 시각화합니다.

    Parameters:
    - video_path: 입력 동영상 파일 경로.
    - YOLO_model_path: YOLO ONNX 모델 파일 경로.
    - YOLO_input_shape: YOLO 모델의 입력 크기 (너비, 높이).
    - HRNet_model_path: HRNet ONNX 모델 파일 경로.
    - HRNet_input_shape: HRNet 모델의 입력 크기 (배치, 채널, 너비, 높이).

    Returns:
    - None. 동영상 프레임을 처리하여 실시간으로 화면에 출력.
    """
    # YOLO ONNX 모델 초기화
    YOLO_session = onnxruntime.InferenceSession(YOLO_model_path)
    YOLO_input_name = YOLO_session.get_inputs()[0].name
    YOLO_output_name = YOLO_session.get_outputs()[0].name

    # HRNet ONNX 모델 초기화
    HRNet_session = onnxruntime.InferenceSession(HRNet_model_path)
    HRNet_input_name = HRNet_session.get_inputs()[0].name
    HRNet_output_name = HRNet_session.get_outputs()[0].name
    b, c, w, h = HRNet_input_shape
    pose_resolution = (w, h)  # 포즈 추정을 위한 입력 이미지 해상도
    resized_pose_resolution = [pose_resolution[0] // 4, pose_resolution[1] // 4]

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        #################### YOLO ####################
        # 전처리: YOLO 모델 입력 형식으로 이미지 변환
        YOLO_input_tensor, scale, scaled_size = preprocess(frame, YOLO_input_shape)

        # YOLO 모델 추론
        outputs = YOLO_session.run([YOLO_output_name], {YOLO_input_name: YOLO_input_tensor})

        # YOLO 후처리: 바운딩 박스 및 클래스 정보 추출
        box_list, scores_list = non_max_suppression(
            outputs, scale_tuple, conf_thres=conf_threshold, iou_thres=iou_threshold
        )
        draw_box_list = []
        pose_box_list = []

        # 탐지 결과 처리
        for box, scores in zip(box_list, scores_list):
            if isinstance(box, (torch.Tensor, list)) and len(box) == 0:
                draw_box_list.append([])
                pose_box_list.append([])
                continue

            # 바운딩 박스 스케일 변환
            box[:, 0:4] = box[:, 0:4] / 640
            scores, classes = scores[:, 0].numpy(), scores[:, 1].numpy().astype(int)
            classes = translate_coco_2017_to_2014(classes)
            draw_box, pose_box = post_process(box.cpu().numpy(), scores, classes, in_width, in_height)
            draw_box_list.append(draw_box)
            pose_box_list.append(pose_box)

        #################### HRNet ####################
        for pose_box in pose_box_list:
            num_of_people = len(pose_box)

            # 사람이 감지되지 않은 경우
            if num_of_people == 0:
                outs = []
            else:
                outs = []
                img = np.array(frame)  # 현재 프레임을 배열로 변환

                # 각 사람의 바운딩 박스를 기반으로 포즈 추정
                for box in pose_box:
                    ########## 전처리 ##########
                    (x1, y1), (x2, y2) = box["pt1"], box["pt2"]

                    # 이미지 자르기 및 리사이즈
                    image_crop, _, _, _, _, _ = crop_image(x1, x2, y1, y2, img, pose_resolution)
                    resized_crop = cv2.resize(image_crop, (pose_resolution[1], pose_resolution[0]))

                    # HRNet 입력 데이터로 변환
                    input_data = transform(resized_crop)
                    HRNet_input_tensor = np.expand_dims(input_data, axis=0).astype(np.float32)

                    ########## 추론 ##########
                    [outputs] = HRNet_session.run([HRNet_output_name], {HRNet_input_name: HRNet_input_tensor})

                    ########## 후처리 ##########
                    outs.append(joints_post_process(outputs[0], box, resized_pose_resolution))

        # 결과 시각화: 바운딩 박스 및 스켈레톤 그리기
        result_frame = draw_boxes_n_skeleton(frame, draw_box_list, pose_box_list, outs)

        end_time = time.time()
        print(f"Frame processed in {(end_time - start_time) * 1000:.2f} ms")

        # 결과 프레임 화면에 표시
        cv2.imshow("YOLO Detection", result_frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 캡처 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()


# 실행 스크립트
if __name__ == "__main__":
    # 동영상 경로 설정
    video_path = "source/le_sserafim.mp4"  # 처리할 동영상 파일 경로

    # YOLO 모델 설정
    YOLO_model_path = "models/yolov7-tiny.onnx"  # YOLO ONNX 모델 파일 경로
    YOLO_input_shape = (640, 640)  # YOLO 모델 입력 크기 (너비, 높이)
    conf_threshold = 0.25  # 객체 탐지 최소 신뢰도 (Confidence Threshold)
    iou_threshold = 0.45  # 바운딩 박스 NMS를 위한 IoU 임계값

    # 입력 동영상의 해상도 읽기
    in_width, in_height = get_video_resolution(video_path)  # 동영상의 너비와 높이 가져오기
    model_input_w, model_input_h = YOLO_input_shape  # YOLO 모델의 입력 크기
    scale = min(model_input_w / in_width, model_input_h / in_height)  # 스케일링 비율 계산
    scaled_w = int(in_width * scale)  # 스케일 조정된 너비
    scaled_h = int(in_height * scale)  # 스케일 조정된 높이
    scale_tuple = (scale, scaled_w, scaled_h)  # YOLO 후처리에 사용할 스케일 관련 정보

    # HRNet 모델 설정
    HRNet_model_path = "models/hrnet.onnx"  # HRNet ONNX 모델 파일 경로
    HRNet_input_shape = (1, 3, 256, 192)  # HRNet 모델의 입력 크기 (배치, 채널, 높이, 너비)

    # 동영상 처리 시작
    process_video(video_path, YOLO_model_path, YOLO_input_shape, HRNet_model_path, HRNet_input_shape)

