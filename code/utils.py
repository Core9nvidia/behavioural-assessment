import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib

from tensorflow.keras.models import load_model

model = load_model('../models/my_model.h5')

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="../models/deploy.prototxt.txt",
                                            caffeModel="../models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

index_to_emotion = joblib.load("../models/id_to_emotion.pkl")
emotions = list(index_to_emotion.values())

emoji_arrs = {emot:cv2.imread(f"../emojis/{emot}.png", cv2.IMREAD_UNCHANGED) for emot in emotions}
emoji_arrs = {k: cv2.resize(v, (50, 50)) for k,v in emoji_arrs.items()}



def detect_faces(image, min_confidence=0.5):

    image_height, image_width, _ = image.shape

    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
    
    results = opencv_dnn_model.forward()    

    boxes = []
    for face in results[0][0]:
        
        face_confidence = face[2]
        
        if face_confidence > min_confidence:

            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)
            boxes.append((x1, y1, x2-x1, y2-y1))
    
    return boxes



# resizing the image
def predict_face(img):
    try:
        image = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(str(e))

    image = image/255.
    img_fed = np.expand_dims(image, axis=0)
    scores = model.predict(img_fed)
    # print(scores)
    index = np.argmax(scores)

    return index_to_emotion[index]


class StatsStore:
    def __init__(self, indict):
        self.dic = indict
    def update(self, emot):
        self.dic[emot] += 1
    def reset(self):
        self.dic = {emot:0 for emot in list(self.dic.keys())}
    

def overlay_emoji(img, pt, emotion):
    x, y = pt
    emoji_img = emoji_arrs[emotion]

    tp_mask = emoji_img[:, :, 3] != 255
    emoji_img[tp_mask, :3] = 0

    img[y:y+emoji_img.shape[0], x:x+emoji_img.shape[1], :] = emoji_img[:, :, :3]
    return img