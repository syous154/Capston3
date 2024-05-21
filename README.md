# Capstone 3 CCTV 영상 내 개인정보 비식별화

* Yolo v8 모델과 [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) 모델을 합쳐서 만듦

* 이번 모델에서 사용하는 가중치 파일 또한 Self-correction-Human-Parsing에서 제공하는 가중치 파일을 사용
  Pascal-Person-Part** ([exp-schp-201908270938-pascal-person-part.pth](https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view?usp=sharing))

## model 사용법

  ```
  python Yolo_Parsing_model.py --parts [원하는 부위] --weights exp-schp-201908270938-pascal-person-part.pth --rate [모자이크 강도] --input-video [input video 경로] --output-video [output-video 경로]
  ```
