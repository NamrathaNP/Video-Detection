from flask import Flask, render_template, Response, request, redirect,url_for
# from camera import VideoCamera
import cv2
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from datetime import datetime
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import argparse
import imutils
import time
import cv2

import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

ct = CentroidTracker()
trackers = []
trackableObjects = {}
(H, W) = (None, None)
# initialize the video writer (we'll instantiate later if need be)
writer = None
now = datetime.now() #video filename
# init gender parameters 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']
totalMale = 0
totalFemale = 0
#hourly_count_dict_male = dict((x,0) for x in np.arange(24))
#hourly_count_dict_female = dict((x,0) for x in np.arange(24))
hourly_count_dict_male = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0}
hourly_count_dict_female = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0}

# face detection confidence threshold
confidence = 0.5
# load our serialized model from disk
print("[INFO] loading models...")
face_net = cv2.dnn.readNetFromCaffe(
                        "face_model/deploy_face.prototxt", 
                        "face_model/face_net.caffemodel")

gender_net = cv2.dnn.readNetFromCaffe(
                        "gender_model/deploy_gender.prototxt", 
                        "gender_model/gender_net.caffemodel")



class VideoCamera(object):
    def __init__(self, cameraid):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # cam= int(cameraid)
        self.video = cv2.VideoCapture(cameraid)
        print(type(cameraid))
        print(cameraid)
        #video capture from IP webcam
        #cap = cv2.VideoCapture('rtsp://admin:admin@172.16.0.14')
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global totalMale, totalFemale, hourly_count_dict_male, hourly_count_dict_female
        (H, W) = (None, None)
        writer = None
        now = datetime.now() #video filename
        # init gender parameters 
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        gender_list = ['Male', 'Female']
        # totalMale = 0
        # totalFemale = 0
        # hourly_count_dict_male = dict((x,0) for x in np.arange(24))
        # hourly_count_dict_female = dict((x,0) for x in np.arange(24))
        # face detection confidence threshold
        confidence = 0.5
        # load our serialized model from disk
        
        face_net = cv2.dnn.readNetFromCaffe(
                                "face_model/deploy_face.prototxt", 
                                "face_model/face_net.caffemodel")

        gender_net = cv2.dnn.readNetFromCaffe(
                                "gender_model/deploy_gender.prototxt", 
                                "gender_model/gender_net.caffemodel")

        success, frame = self.video.read()
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # cv2.imwrite('./static/test.jpg',frame)

        #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = imutils.resize(frame, width=400)

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('./video_archive/'+now.strftime('%d-%m-%Y_%H-%M-%S')+'.avi', fourcc, 24,(W, H), True)

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")

                if(startX<0 or startY<0):
                    break

                face_img = frame[startY:endY, startX:endX].copy()
                if not face_img.size:
                    # print('breaking from face')
                    break

                # blob2 = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)


                # Predict gender
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                overlay_text = "%s" % (gender)
                cv2.putText(frame, overlay_text ,(startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)

        cv2.line(frame, (50, 0), (50, H), (0, 255, 255), 1)
        cv2.line(frame, (350, 0), (350, H), (0, 255, 255), 1)          
        save_frame = frame.copy()       
        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the x-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'left' and positive for 'right')
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)
                to.centroids.append(centroid)

                        # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative/positive (indicating the object
                # is moving left/right) AND the centroid is beyond the left/right
                # line, count the object
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)
                to.centroids.append(centroid)

                if (direction < 0 and centroid[0] <=50) or (direction>0 and centroid[0] >=350):
                    now = datetime.now()
                    time = str(now.hour)
                   
                    if gender is 'Male':
                        totalMale += 1
                        to.counted = True
                        
                        hourly_count_dict_male[time]+=1
                        
                        print('Male:\n',hourly_count_dict_male)
                       
                        (pd.DataFrame.from_dict(data=hourly_count_dict_male, 
                        orient='index').to_csv('./static/male.csv', header=True))
                        socketio.sleep(1)
                        socketio.emit('newmale', {'hourly_count_dict_male': hourly_count_dict_male, 'totalMale': totalMale} , namespace='/test')

                        # print("HI")

                    elif gender is 'Female':
                        totalFemale += 1
                        to.counted = True
                        
                        hourly_count_dict_female[time]+=1

                        print('Female:\n',hourly_count_dict_female)
                        (pd.DataFrame.from_dict(data=hourly_count_dict_female, 
                        orient='index').to_csv('./static/female.csv', header=True))
                        socketio.sleep(1)
                        socketio.emit('newfemale', {'hourly_count_dict_female': hourly_count_dict_female,'totalFemale':totalFemale} , namespace='/test')

                        # print("HI")

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to


            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        # info = [
        #     ("Male", totalMale),
        #     ("Female", totalFemale),
        #     ("Total Count", totalMale+totalFemale)
        # ]   

        # ## loop over the info tuples and draw them on our frame
        # for (i, (k, v)) in enumerate(info):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        # return '{} {} {}'.format(jpeg.tobytes(), hourly_count_dict_male, hourly_count_dict_female)


@app.route('/')
def index():
    # totalMale = 6
    # totalFemale = 7
    # hourly_count_dict_male = {0: 0, 1: 0, 2: 0, 3: 2, 4: 4, 5: 4, 6: 2, 7: 10, 8: 1, 9: 8, 10: 4, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
    # hourly_count_dict_female = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 5, 6: 7, 7: 6, 8: 4, 9: 1, 10: 6, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
    # # print(totalMale)
    # socketio.emit('newframe', {'hourly_count_dict_male': hourly_count_dict_male, 'hourly_count_dict_female': hourly_count_dict_female,'totalMale':totalMale, 'totalFemale':totalFemale} , namespace='/test')

    return render_template('main.html', hourly_count_dict_male=hourly_count_dict_male, hourly_count_dict_female=hourly_count_dict_female, totalMale=totalMale, totalFemale=totalFemale)

@app.route('/camera_select', methods=["GET", "POST"])
def cameraidlookup():
    cameraid = request.form.get('cameraid')
    print(cameraid);
    # return render_template('video_feed.html', cameraid=cameraid)
    return redirect(url_for('video_feed',  cameraid=cameraid))
    

def gen(camera):
    global ct, trackers,trackableObjects, writer, now, MODEL_MEAN_VALUES,gender_list,totalMale, totalFemale, hourly_count_dict_male, hourly_count_dict_female,confidence, face_net, gender_net

    print("[INFO] starting video stream...")

    while True:
        frame = camera.get_frame()
        # socketio.emit('newnumber', {'display':"HI 4"} , namespace='/test')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed/<int:cameraid>')
def video_feed(cameraid):
    # cameraid = request.form.get('cameraid')
    # print (cameraid);
    return Response(gen(VideoCamera(cameraid)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    # app.run( debug=True)
    socketio.run(app, debug=True)