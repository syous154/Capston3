import torch
import torchvision.transforms as transforms
from utils.transforms import transform_logits
import numpy as np
import cv2
from utils.transforms import get_affine_transform
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import networks             

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Yolo + Human Parsing in video")

    parser.add_argument("--parts", type=str, nargs='+', default=[], choices=['Head', 'Torso', 'Upper_Arms', 'Lower_Arms', 'Upper_Legs', 'Lower_Legs'])
    parser.add_argument("--weights", type=str, default='', help="pre_trained model weights")
    parser.add_argument("--rate", type=int, default=5, help="Set the degree of mosaic.")
    parser.add_argument("--input-video", type=str, default='', help="path of input video.")
    parser.add_argument("--output-video", type=str, default='', help="path of output video.")

    return parser.parse_args()


def human_parsing(frame, weights, rate, parts):
    # pascal 데이터 셋 기준
    num_classes = 7
    input_size = [512, 512] 
    label = ['Background', 'Head', 'Torso', 'Upper_Arms', 'Lower_Arms', 'Upper_Legs', 'Lower_Legs']
    
    ## 모델 backbone 불러오기
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    
    ## pre-trained 가중치 적용
    state_dict = torch.load(weights)['state_dict']
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    
    original_frame = frame
    blur_frame = frame
    torso_frame = frame
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    
    aspect_ratio = input_size[1] * 1.0 / input_size[0]  ## 이미지의 종횡비를 계산
    h, w, _ = frame.shape                               ## 입력 이미지의 높이와 너비
    center = np.zeros((2), dtype=np.float32)            ## 이미지의 중심 좌표를 저장할 배열을 생성
    
    center[0] = 0 + (w-1) * 0.5                         ## 중심좌표 계산
    center[1] = 0 + (h-1) * 0.5
    
    if w > aspect_ratio * h:                            ## 이미지의 가로세로 비율이 주어진 input_size의 비율과 다를 경우, 이미지의 크기를 조정합니다.
        h = w * 1.0 /aspect_ratio                       ## 세로의 경우 가로의 비율로 조정하고, 가로의 경우 세로의 비율로 조정합니다.
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
        
    scale = np.array([w, h], dtype=np.float32)          ## 이미지의 크기 및 변환을 적용하기 위해 변환 행렬을 계산합니다. 
    c = center                                          ## 이때 get_affine_transform 함수가 사용됩니다. 이 함수는 이미지의 중심, 크기, 회전 등을 고려하여 이미지 변환에 필요한 행렬을 생성
    s = scale    
    r = 0
    trans = get_affine_transform(c, s, r, input_size)
    
    frame = cv2.warpAffine(                             ## 이미지에 변환 행렬을 적용하여 이미지를 변환, 이를 통해 이미지가 모델에 입력으로 사용될 크기로 조정
            frame,
            trans,
            (int(input_size[1]), int(input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
    
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    
    
    with torch.no_grad():

        output = model(frame.cuda())   # output은 list 형태
        upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)  ## 세그멘테이션 맵의 크기를 원래 크기로 업샘플링합니다. `torch.nn.Upsample`을 사용하여 bilinear 보간을 통해 업샘플링합니다. 
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)     ## 세그멘테이션 맵을 변환합니다. `transform_logits` 함수를 사용하여 세그멘테이션 맵을 입력 이미지와 동일한 크기로 변환합니다.
        parsing_result = np.argmax(logits_result, axis=2)       ## 로짓 결과에서 클래스에 대한 가장 높은 확률을 가진 인덱스를 가져와 세그멘테이션 결과를 얻습니다.
                                                                ## parsing_result는 세그멘테이션 맵으로, 각 픽셀에 해당하는 클래스 인덱스를 담고 있는 2차원 배열입니다. 여기서 각 픽셀의 값은 클래스를 나타내는 정수 값입니다.
        class_of_interest = [3,4,5,6]

        rate = 5           # 모자이크 처리에 사용할 축소 비율 (1 / rate)
        w = len(blur_frame[0])
        h = len(blur_frame)
        blur_frame = cv2.resize(blur_frame,(max(w // rate, 1), max(h // rate, 1))) # rate 비율을 사용해 축소
                
        blur_frame = cv2.resize(blur_frame, (w,h), interpolation=cv2.INTER_AREA)
                
        for i in range(len(blur_frame)):
            for j in range(len(blur_frame[0])):
                if parsing_result[i][j] in class_of_interest:
                    continue
                else:
                    blur_frame[i][j] = original_frame[i][j]
            
    return blur_frame


def main():
    
    args = get_arguments()
    # 동영상 파일 열기
    cap = cv2.VideoCapture(args.input_video)

    # 동영상의 FPS와 해상도 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 결과를 저장할 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model 
    model.cuda()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상의 총 프레임 수 가져오기
    
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if ret:
            results = model(frame)
            nms_human = sum(results[0].boxes.cls == 0)  ## 프레임에 사람이 몇명있는가
            if nms_human > 0:                           ## 프레임 내에 사람이 있다면
                for bbox in results[0].boxes:
                    if bbox.cls.item() == 0:            ## 객체 탐지한 것 중 사람만
                        start_point = (int(bbox.xyxy[0][0].item()), int(bbox.xyxy[0][1].item()))
                        end_point = (int(bbox.xyxy[0][2].item()), int(bbox.xyxy[0][3].item()))

                        # 휴먼파싱 후 블러처리 까지 하는 함수 호출
                        blur_frame = human_parsing(frame[ start_point[1] : end_point[1], start_point[0] : end_point[0]], args.weights, args.rate, args.parts)
                        
                        frame[ start_point[1] : end_point[1], start_point[0] : end_point[0]] = blur_frame
                # 결과 동영상 파일에 프레임 쓰기    
            out.write(frame)
        else:
            break

    # 동영상 파일 닫기
    cap.release()
    out.release()

if __name__ == '__main__':
    main()