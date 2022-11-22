import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from utils import detect_faces, predict_face, StatsStore, emotions, overlay_emoji

import matplotlib.pyplot as plt
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)
switch = False
rec = False
emoji = False
stats = StatsStore({emot:0 for emot in emotions})


plot = False
cv2.imwrite("static/plotimg.png", np.zeros((1,1,3)))

# text to speech
import pyttsx3



def gen_frames():
    global switch, camera, rec, stats, plot, emoji
    first = True

    while True:
        ret, frame = camera.read()

        if not switch:
            frame = np.ones((1, 1, 3))

        if rec:
            frame = cv2.flip(frame, 1)

            faces = detect_faces(frame)

            if len(faces):
                for (x, y, w, h) in faces:
                    try:
                        # frame = cv2.rectangle(frame, (x-5, y-10), (x+w+5, y + h+10), (255, 0, 0), 2)

                        roi_color = frame[y+25:y-10 + h, x+20:x + w-20, :]
                        gray_single = np.squeeze(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY))
                        roi_gray = np.stack([gray_single,]*3, axis=-1)

                        # resizing the image
                        emotion = predict_face(roi_gray)
                        stats.update(emotion)

                        # frame = cv2.putText(frame, emotion, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), thickness=2)
                        if emoji:
                            frame = overlay_emoji(frame, (x, y-15), emotion)
                            
                    except:
                        pass

            # writing stats
            d = 350
            for emot in stats.dic.keys():
                frame = cv2.putText(frame, f"{emot} : {stats.dic[emot]}", (440, d), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,255), thickness=2)
                d += 25
            d = 350

            if plot:
                plt.clf()
                plt.figure(facecolor='#94F008')
                plt.bar(list(stats.dic.keys()), list(stats.dic.values()), width=0.4, color='red');
                plt.title("Emotion Stats")
                plt.xlabel("Emotions")
                plt.ylabel("Face count")

                plt.yticks(np.arange(0, max(list(stats.dic.values()))+2))

                plt.savefig("static/plotimg.png")

            stats.reset()

            
            # flip it back
            frame = cv2.flip(frame, 1)


        if ret:
            try:
                _, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            break


@app.route('/')
def Start():
    """ Home Page """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """ A route that returns a streamed response needs to return a Response object
    that is initialized with the generator function."""

    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def speak_emotion():
    global camera
    frames = []
    for i in range(1):
        frames.append(camera.read()[1])

    emotions = []
    for frame in frames:
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            try:

                roi_color = frame[y+25:y-10 + h, x+20:x + w-20, :]
                gray_single = np.squeeze(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY))
                roi_gray = np.stack([gray_single,]*3, axis=-1)

                # resizing the image
                emotion = predict_face(roi_gray)

                emotions.append(emotion)
            except:
                pass

    print(emotions)
    max_emo = max(set(emotions), key=emotions.count)
    print(f"\n\n\nMain emotion is {max_emo}\n\n")

    engine = pyttsx3.init()
    engine.say(f"The emotion of the person in front is {max_emo}")
    engine.runAndWait()
    engine = None


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera, switch, rec, plot, emoji

    if request.method == 'POST':
        
        if request.form.get('stop') == 'Start/Stop':
            switch = not switch

        if request.form.get('rec_button') == 'Recognize':
            rec = not rec

        if request.form.get('plot') == 'Refresh Plot':
            plot = not plot
            time.sleep(2)
            plot = not plot

        if request.form.get('emoji_button') == 'Show Emoji':
            emoji = not emoji

        if request.form.get('audio') == 'Speak':
            speak_emotion()

            

    elif request.method=='GET':
        return render_template('index.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)



camera.release()
cv2.destroyAllWindows()   