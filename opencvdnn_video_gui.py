import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
#모델의 이름
prototxt_name = 'deploy.prototxt'
#모델의 설계도
#2개의 학습된 파일을 이용할 것!
min_confidence = 0.5
#최소의 확률
title_name = 'dnn Deep Learning object detection Video'
frame_width = 300
frame_height = 300
file_name ='./image/make.mp4'
cap = cv2.VideoCapture()

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./", title = "Select file", filetypes = (("mp4 files", "*.mp4"),("all files", "*.*")))
    global cap #전역변수를 통해서 변화를 전역변수 cap에 전달하기 위해서
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()
    
def detectAndDisplay():
    _, frame = cap.read() #자체적으로 계속 무한적으로 돌도록!
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    ############ 프레임 크기 조정하기
    (h, w) = frame.shape[:2]
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    #규칙 : readNetFromCaffe는 prototxt 와 model(caffemodel)을 가져와야된다
    #그것을 model이라고 설정
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    #모델이 인식할 blob으로 만들기 위해서 300x300 사이즈로 만든다
    #cv2.dnn.blobFromImage(이미지자체, 각 픽셀 값의 배율, 네트워크에 대한 기본입력은 320x320이다, 훈련시키는 동안 사용되었던 각 이미지에서 빼야 하는 평균)
    #이미지를 블롭으로 바꾼다
    #블롭 :    블롭은 Mat 타입의 4차원 행렬로 표현됩니다.
    #이때 각 차원은 NCHW 정보를 표현합니다.
    #N : 영상개수, C: 채널개수, H, W: 영상의 세로와 가로 크기
    
    model.setInput(blob)
    #이전에 설정한 model에 blob을 입력으로 넣는다
    detections = model.forward()
    #얼굴을 인식한다!

    min_confidence = float(sizeSpin.get())
    #sizeSpin으로 통해 가져온 값을 min_confidence에 넣는다

    for i in range(0, detections.shape[2]):
        #얼굴 인식된 개수 만큼 반복!
        confidence = detections[0, 0, i, 2]
        #얼굴 인식된 녀석!

        if confidence > min_confidence:
            #얼굴 인식된 녀석 중에 min_confidence보다 높은 녀석만
            (height, width) = frame.shape[:2]
            #동영상 하나의 프레임의 높이와 너비를 구한다
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #box라는 녀석 안에 찾은 얼굴을
            #3->시작가로, 4->시작세로, 5->끝가로, 6->끝세로를 (각각 분수로 되어있는 것 같음)
            #np.array와 곱해서 값을 box에 넣는다
            (startX, startY, endX, endY) = box.astype("int")
            #박스 안에는 4개의 값이 있는데 각각의 값을 변수에 만들어서 넣는다

            text = "{:.2f}".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detectAndDisplay)
            
main = Tk()
main.title(title_name)
main.geometry()

label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
sizeLabel = Label(main, text="Min Confidence : ")
sizeLabel.grid(row=1, column=0)
sizeVal = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal, from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=0, columnspan=4)

Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2)
imageFrame = Frame(main)
imageFrame.grid(row=2, column=0, columnspan=4)

lmain = Label(imageFrame)
lmain.grid(row=0, column=0)



