import cv2
import face_recognition
import pickle
import time

image_file = 'image_detect/jyeon2.jpg'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'

model_method = 'cnn'

def detectAndDisplay(image):
    start_time = time.time()
    #인식하는데 얼마나 걸리는지 시간 측정
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #RGB로 변환!

    boxes = face_recognition.face_locations(rgb, model=model_method)
    #이미지 안에 모든 얼굴을 찾기 위해서 가져옴!
    encodings = face_recognition.face_encodings(rgb, boxes)
    #찾은 얼굴의 특성들을 128가지의 특성으로 encodings에 넣는다
   
    
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        #가져온 것을 딕셔너리로 우리가 pickle로 가져온 data의 것과 비교를 한다!
        name = unknown_name

        if True in matches: #만약에 son 혹은 tedy의 값에 1개라도 충족하는 얼굴이 있으면!
            matchedIdxs = [i for (i, b) in enumerate(matches) if b] #matches 안에는 순서대로 비교해서 가져온 값이 들어 있다
            print(matchedIdxs) #비교되서 True 인 곳의 indexfmf enumerate를 통해서 matchedIdxs로 넣는다
            print(matches)
            counts = {} #counts라는 딕셔너리를 만든다

            for i in matchedIdxs:
                name = data["names"][i] #data의 이름을 name으로 가져와서 
                counts[name] = counts.get(name, 0) + 1
                #counts딕셔너리에 key가 name이라는 녀석이 없어도 키의 name이 생성되고, 기본값 0이 들어간다,
                #만약 값이 있으면 값 + 1을 해서 1개를 더한것을 counts[name] 값에 저장한다 a += 1 같은 느낌
                #얼굴의 키값이 이름인 개수의 값을 가져오기 위해서 있으면 +1 해서 더한다

            name = max(counts, key=counts.get)
            #counts 딕셔너리에서 기준을 counts.get을 이용해서 value의 값중 가장 큰것을 name으로 설정
            #가장 많이 나온 이름을 이름으로 쓰겠다!

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        #찾은 얼굴에 name도 들어가서 boxes 정보는 top, right, bottom, left 위치를 나타내고
        #name은 이름이 들어간다
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if (name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
        #인식한 곳에 그림을 그리고 텍스트를 쓴다!


        end_time = time.time()
        process_time = start_time - end_time
        #걸린 시간을 확인하기 위해서 설정

        print("{:.3f} second".format(process_time))
        cv2.imshow("Recognition", image)
    

data = pickle.loads(open(encoding_file, "rb").read())
#data라는 곳에 pickle.load를 통해서 .pickle을 불러온다


image = cv2.imread(image_file)
#이미지를 읽어온다
detectAndDisplay(image)
#함수로 만들기 위해서 함수에 이미지를 넣는다

cv2.waitKey(0)
cv2.destroyAllWindows()
#아무키가 눌리면 창이 닫아진다
