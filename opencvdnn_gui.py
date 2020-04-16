import cv2
import sys, os 
import numpy as np
import unicodedata
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from tkinter import messagebox

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
#모델의 이름
prototxt_name = 'deploy.prototxt'
#모델의 설계도
#2개의 학습된 파일을 이용할 것!
min_confidence = 0.5
#최소의 확률
#file_name ='WIN_20190408_17_12_58_Pro.jpg'

title_name = 'dnn Deep Learning object detection'
frame_width = 300
frame_height = 300

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./", title = "Select file", filetypes = (("jpeg files", "*.jpg"),("png files", "*.png"),("all files", "*.*")))
    print('File name : ', file_name)

    ############ 엄청 중요 ############### imread는 한글을 읽을 수 없음!!
    if file_name: #한글 폴더 및 한글이 있으면 안되서 그것을 byte로 만들어서 numpy를 통해서 코드를 바꿈!
        file_name = open(file_name, 'rb')
        bytes = bytearray(file_name.read()) 
        numpyarray = np.asarray(bytes, dtype=np.uint8) #이거 해서 사진 돌아가는 경우 있음!
        read_image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    #read_image = cv2.imread(file_name)
        
    try:
        read_image = cv2.resize(read_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    except:
        messagebox.showinfo(title="오류", message="적용이 안되는 사진 다른 사진을 이용하세요.")
        return None
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    print(height, width)
    detectAndDisplay(read_image, width, height)


def detectAndDisplay(frame, w, h):
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
    min_confidence = float(sizeSpin.get())
    ###여기서 얼굴을 찾음!!!

    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] #여기 안에 들어있는 confidence 값을 가져온다
        #confidence 확률!

        if confidence > min_confidence:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
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

    #Tkinter에 보여주기 때문에
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image =imgtk

            
main = Tk()
main.title(title_name)
main.geometry()

#read_image = cv2.imread(None)
#read_image = cv2.resize(read_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
#image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
#image = Image.fromarray(image)
#imgtk = ImageTk.PhotoImage(image=image)
#(height, width) = read_image.shape[:2]

label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row = 0, column = 0, columnspan = 4)
sizeLabel = Label(main, text = 'Min Confindence : ')
sizeLabel.grid(row = 1, column = 0)
sizeVal = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal, from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main, text = "File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2)
detection = Label(main, image=None)
detection.grid(row = 2, column = 0, columnspan = 4)
#detectAndDisplay(read_image, width, height)

main.mainloop()
