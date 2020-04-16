import cv2
import face_recognition
import pickle
import time

file_name = 'video/facedetect.mp4'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'

model_method = 'hog'
output_name = 'video/output_' + model_method + '.avi'

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
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if (name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''

        cv2. rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)

    image = cv2.resize(image, None, fx = 0.5, fy = 0.5)
    cv2.imshow("Recognition", image)

    global writer
    #함수 안에 있는 writer여서 선언을 해줘야 된다!
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24, (image.shape[1], image.shape[0]), True)

    if writer is not None:
        writer.write(image)

    

data = pickle.loads(open(encoding_file, "rb").read())

cap = cv2.VideoCapture(file_name)
writer = None

if not cap.isOpened:
    print("오류")
    exit(0)

while True:
    ret, frame = cap.read()
    
    if frame is None:
        print("끝")
        cap.release()
        writer.release()
        break
    
    detectAndDisplay(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        writer.release()
        break
        cv2.destroyAllWindows()


        


