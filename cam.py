import numpy as np
import cv2
import pickle

recognizer=cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("C:/Users/ANKIT RAJ/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# Define the codec and create VideoWriter object
recognizer.read("trainner.yml")
labels ={"person_name": 1}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        # write the flipped frame

        for x, y, w, h in faces:
            roi_gray=gray[y:y+h, x:x+w]
            roi_color=frame[y:y+h, x:x+w]

            id_, conf=recognizer.predict(roi_gray)
            if conf>=45:
                print(id_)
                print(labels[id_])
                font= cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                cv2.putText(frame,name,(x,y),font,1,(255,255,255),2)
          #  cv2.imwrite(img_item, roi_gray)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()

cv2.destroyAllWindows()