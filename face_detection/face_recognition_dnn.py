import cv2
import face_recognition
import pickle

dataset_paths = ['dataset/son/', 'dataset/tedy/']
#데이터셋 폴더 경로를 리스트에 넣음'
names = ['Son', 'Tedy']
#이름을 넣음
number_images = 5
#학습할 이미지 숫자를 넣음
image_type = '.jpg'
#이미지의 확장자를 넣음
encoding_file = 'encodings.pickle'
#인코딩된 파일 이름을 설정
model_method = 'cnn'
#모델의 방식은 cnn이라고 설정
#정확도가 좋지만 속도가 느리다...
#HOG 방식이 빠르다!! 하지만 정확도가 낮다

knownEncodings = []
knownNames = []
#2가지 배열을 만듬

for (i, dataset_path) in enumerate(dataset_paths):
    #enumerate()를 이용하면 인덱스 값까지 가져올 수 있다!
    #i 값에는 인덱스 값! dataset_path에는 dataset_paths 값!!
    name = names[i]
    #반복문에서 dataset_path가 son일경우 names[0]이되어서 names에는 Son을 가져온다

    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type
        #폴더안에있는 파일 이름을 인식하기 위해서!!
        #number_images 즉 range(10)이므로 idx는 0부터
        #즉 dataset/son/1.jpg 이 된다!\

        image = cv2.imread(file_name)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model=model_method)
        #face_recognition을 이용해서 model을 cnn 방식으로 해당 이미지의 얼굴만 찾아서 위치를 boxes로 저장한다
        #boxes 값 : [ (top(y축맨위), right(x축맨오른), bottom(y축맨밑), left(x축맨 왼쪽) ) ] 리스트안에 튜플로

        encodings = face_recognition.face_encodings(rgb, boxes)
        #원본 이미지와 원본이미지에서 boxes로 가져온 해당 얼굴을 인코딩 시켜준다!

        for encoding in encodings:
            #총 128개의 값이 나와서 반복
            print(file_name, name, encoding)
            #128개의 추출 값을 보기 위해서
            knownEncodings.append(encoding)
            #특성을 얻은 값을 knownEncodings의 배열안에 넣는다
            knownNames.append(name)
            #또한 같이 이름을 넣는다

data = {"encodings":knownEncodings, "names":knownNames}
#해당 리스트 들을 data라는 딕셔너리에 넣는다
f = open(encoding_file, "wb")
#변수 encoding_file에 해당되는 파일을 만들어서 bit 형식으로 값을 저장하기 위해서 wb설정후 연다
f.write(pickle.dumps(data))
#파일 객체에 pickle.dumps를 이용해서 해당 값을 encoding_file에 넣는다!
f.close()
#닫기
