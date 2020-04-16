import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

face_cascade_name = 'opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
file_name = 'image/facedetect.mp4'
title_name = 'Haar cascade object detection'
frame_width = 500
cap = cv2.VideoCapture() #빈 비디오 capture를 cap에 저장

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

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./image", title = "Select file", filetype = (("MP4 files", "*.mp4"), ("all files", "*.*")))
    print('File name : ', file_name)
    global cap #global로 설정하지 않으면 함수를 실행해도 덮어쓰지 않는다! 
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()

def detectAndDisplay():
    _, frame = cap.read() #_의 의미는 값을 해당 부분의 값을 무시한다는 의미
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

    (height, width) = frame.shape[:2]
    frameSize = int(sizeSpin.get())
    ratio = frameSize / width
    dimension = (frameSize, int(height * ratio))
    frame = cv2.resize(frame, dimension, interpolation = cv2.INTER_AREA)
    #frame 크기 변경 까지 하려면 이 소스 필요
    
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) #RGB로 바꿔 줘라
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image = imgtk)
    lmain.after(10, detectAndDisplay)
    #밑에서 만든 frame이라는 곳에 보여주는 것으로 변경

        #단계 : 1. 회색화, 2. 또 필터, 3.얼굴을 찾아줘!, 4. 찾은 얼굴들 반복문 돌려서 위치를 가져와 표현(사각형)
        #5. 찾은 얼굴안에서 눈을 찾기 위해서 그 부분만 뽑아냄
        #6. 또 다시 얼굴안에 있는 눈을 찾아줘!
        #7. 찾은 눈을 표현
        #8. 표현한 얼굴과 눈을 보여줘!

#tkinter은 대표적인 GUI를 만들어줄 수 있는 파이썬 방식
#main
main = Tk()
main.title(title_name)
main.geometry()

label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
sizeLabel = Label(main, text='Frame Width : ')
sizeLabel.grid(row=1, column=0)
sizeVal = IntVar(value=frame_width)
sizeSpin = Spinbox(main, textvariable=sizeVal, from_=0, to=2000, increment=100, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2)

imageFrame = Frame(main) #프레임을 보여줄 수 있도록 설정!
imageFrame.grid(row=2, column=0, columnspan=4)

lmain = Label(imageFrame) #하나의 Label 공간을 만들어서 그곳에 imageFrame 즉 비디오를 띄움
lmain.grid(row=0, column=0)

main.mainloop() #GUI 시작

