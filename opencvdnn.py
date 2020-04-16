import cv2
import numpy as np

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
#모델의 이름
prototxt_name = 'opencv/samples/dnn/face_detector/deploy.prototxt'
#모델의 설계도
#2개의 학습된 파일을 이용할 것!
min_confidence = 0.5
#최소의 확률
file_name ='WIN_20190408_17_12_58_Pro.jpg'

def detectAndDisplay(frame):
    #blob 형태로 변환, 모델을 지정,
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    #opencv에 있는 dnn을 사용
    #readNetFromCaffe Caffe모델을 사용한다

    #넣을 때는 사이즈를 바꾼다!
    #frame으로 받아온 그림을 300에 300 사이즈로 바꾼다
    #나머지는 nomalize하는 값들을 넣는다 ( 1.0, (300, 300), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob) #blob라는 녀석을 입력값으로 넣는다
    detections = model.forward() #입력되서 찾은 녀석들을 detections에 넣는다 전부
    ###여기서 얼굴을 찾음!!!

    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] #여기 안에 들어있는 confidence 값을 가져온다
        #confidence 확률!

        if confidence > min_confidence:
            box = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int") #곱한 값이 실수로 될수도 있어서 정수로 만듦
            #3:7로 되어있는게 아마 startX, startY로 설정이 되어있을 것이다
            print(confidence, startX, startY, endX, endY)
            #값 보기 위해서
            text = "{:.2f}%".format(confidence * 100) #확률을 보여주기 위해서! text 변수에 넣음
            y = startY - 10 if startY - 10 > 10 else startY + 10 #글의 위치를 만들기 위해서 위에 공간이 없으면 밑에 나오기
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            #얼굴부위에 네모들 만들기
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #얼굴부위에 네모에 글을 적기 위해서 씀

    cv2.imshow("detected", frame)

            
print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread(file_name)
img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
cv2.imshow("PRE",img)

(height, width) = img.shape[:2]

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
