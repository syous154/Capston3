# Capstone 3 CCTV 영상 내 개인정보 비식별화

* Yolo v8 모델과 [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) 모델을 합쳐서 만듦

* 이번 모델에서 사용하는 가중치 파일 또한 Self-correction-Human-Parsing에서 제공하는 가중치 파일을 사용

  * Pascal-Person-Part ([exp-schp-201908270938-pascal-person-part.pth](https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view?usp=sharing))

    labels = [ 'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs' ]

  * mIoU on Pascal-Person-Part validation: **71.46** %.
 
    
## Yolo_Parsing_model 사용법

  ```
  python Yolo_Parsing_model.py --parts [원하는 부위] --weights exp-schp-201908270938-pascal-person-part.pth --rate [모자이크 강도] --input-video [input video 경로] --output-video [output-video 경로]
  ```


## Web으로 실행하는 방법

```
 streamlit run app.py
```
위 코드를 실행하면 로컬로 웹 사이트를 열림. 원한다면 streamlit내 기능을 통해 배포까지도 가능
웹 사이트가 열리면 동영상 파일을 업로드 후 모자이크할 신체부위 모자이크 강도를 선택하면 동영상을 처리 후 다운로드 가능
