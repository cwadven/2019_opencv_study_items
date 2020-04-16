import cv2
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #색깔이 있으면 오류가 날수 있어서 Gray로
    drame_gray = cv2.equalizeHist(frame_gray) #히스토그램을 통해서 디지털 이미지 처럼 단순화.. 나중
    faces = face_cascade.detectMultiScale(frame_gray) #쪼개져있는 그림을 만들어 준다 얼굴이면 얼굴

    for (x, y, w, h) in faces: #x -> 시작 x축 좌표, y -> 시작 y축 좌표, w -> 너비, h -> 높이
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        #cv2.rectangle(사진, 왼쪽위, 오른쪽밑, 색깔, 두께)
        faceROI = frame_gray[y:y+h, x:x+w]
        #ROI:관심있는 구역이라고 보면됨
        #방금 꺼낸 얼굴만 가져온다! 반대로 왜 했는지는 모르겠음!

        eyes = eyes_cascade.detectMultiScale(faceROI)
        
        for (x2, y2, w2, h2) in eyes: #x2 -> 시작 x축 좌표, y2 -> 시작 y축 좌표, w2 -> 너비, h2 -> 높이
            eye_center = (x + x2 + w2//2, y + y2 + h2//2) #좌표를 찾기 위해서 기존 + 추가적으로 한 부분
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
            #cv2.circle(사진, 중앙 좌표, 반지름, 색깔, 두께)
        cv2.imshow('Capture - Face detection', frame)

        #단계 : 1. 회색화, 2. 또 필터, 3.얼굴을 찾아줘!, 4. 찾은 얼굴들 반복문 돌려서 위치를 가져와 표현(사각형)
        #5. 찾은 얼굴안에서 눈을 찾기 위해서 그 부분만 뽑아냄
        #6. 또 다시 얼굴안에 있는 눈을 찾아줘!
        #7. 찾은 눈을 표현
        #8. 표현한 얼굴과 눈을 보여줘!
            
print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread("WIN_20190408_17_12_58_Pro.jpg")
img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

print("width: {} pixcels".format(img.shape[1]))
print("height: {} pixcels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

face_cascade_name = 'opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
#opencv 에서 제공해주는 xml 파일 cascade를 가져온다!
#수동으로 노가다한 정보를 xml로 때려 박은 것이다!

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
#opencv에서 재공하는 CascadeClassifier라는 함수를 변수로 만든다

#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)): #xml을 가져와서 사용한다
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)): #xml을 가져와서 사용한다
    print('--(!)Error loading eyes cascade')
    exit(0)

detectAndDisplay(img)
