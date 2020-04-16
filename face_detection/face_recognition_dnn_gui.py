import cv2
import face_recognition
import pickle
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import numpy as np

image_file = 'image_detect/changwoo.jpg'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'
title_name = 'Face Recognition'

model_method = 'cnn'

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./", title = "Select image", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    detectAndDisplay(read_image)

def detectAndDisplay(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        y = top - 15 if top - 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if (name == unknown_name):
            color = (255, 0, 0)
            line = 1
            name = ''

        cv2.rectangle(image, (left, top), (right,bottom), color, line)
        y = top - 15 if top - 5 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)

    #인식한 이미지를 imgtk를 통해서 Label안에 들어갈 image에 넣을 imgtk를 만들어야되서!
    image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    #detection의 정보를 수정한다!
    detection.image = imgtk
        
        

main = Tk()
main.title(title_name)
main.geometry()

data = pickle.loads(open(encoding_file, "rb").read())
#학습한 정보를 가져온다

read_image = cv2.imread(image_file)
#이미지를 읽는다

image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
#가져온 이미지를 imgtk에 넣을수 있도록 가공 1. BGR -> RGB로
image = Image.fromarray(image)
#PhotoImage에 있는 image에 넣기 위해서 형식으로 가공 2. 배열 -> 이미지로
imgtk = ImageTk.PhotoImage(image=image)
#ImageTk안에 넣음

(height, width) = read_image.shape[:2]
#읽은 이미지의 높이와 너비를 가져온다

label = Label(main, text=title_name)
#맨위에 이름을 Label(공간)로 설정 및 text를 넣는다
label.config(font=("Courier", 18))
#라벨안의 글꼴을 설정
label.grid(row=0, column=0, columnspan=4)
#그 라벨의 위치를 지정
#1. 라벨 생성 및 안에 글 Label(어떤 Tk에!, text="어쩌고")
#2. 라벨 꾸미기 추가 .config
#3. 라벨의 위치 지정 .grid

Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=0, columnspan=4)
detection = Label(main, image=imgtk)
#Label의 특성에 맞는 image를 넣어야되서 ImageTk를 통해서 받은 값을 가져온다
detection.grid(row=2, column=0, columnspan=4)

detectAndDisplay(read_image)

main.mainloop()
#한번만 아니고 계속 반복
