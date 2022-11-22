import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import detect_faces, predict_face, StatsStore, emotions, overlay_emoji

cap = cv2.VideoCapture(0)


stats = StatsStore({emot:0 for emot in emotions})

ret, frame = cap.read()
while (ret == True):
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    
    # print(frame.shape) # 480,640,3

    faces = detect_faces(frame)

    if len(faces):
        for (x, y, w, h) in faces:
            try:
                frame = cv2.rectangle(frame, (x-5, y-10), (x+w+5, y + h+10), (255, 0, 0), 2)

                roi_color = frame[y+25:y-10 + h, x+20:x + w-20, :]
                gray_single = np.squeeze(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY))
                roi_gray = np.stack([gray_single,]*3, axis=-1)

                # resizing the image
                emotion = predict_face(roi_gray)
                stats.update(emotion)

                # frame = cv2.putText(frame, emotion, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), thickness=2)
                frame = overlay_emoji(frame, (x, y-15), emotion)
            except:
                pass


    d = 350
    for emot in stats.dic.keys():
        frame = cv2.putText(frame, f"{emot} : {stats.dic[emot]}", (440, d), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,255), thickness=2)
        d += 25
    d = 350
    stats.reset()


    cv2.imshow('detection', frame)
    

    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()

