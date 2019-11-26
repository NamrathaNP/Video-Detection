# USAGE
# python main.py

# import the necessary packages
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import cv2
from flask import Flask ,request, Response
from flask import render_template
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from flask_socketio import SocketIO
# from random import randint
from time import sleep
from threading import Thread, Event

# import os
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

thread = Thread()
thread_stop_event = Event()

class RandomThread(Thread):
    def __init__(self):
        self.delay = 2
        super(RandomThread, self).__init__()
# @app.route('/')
# def index():
#     return render_template('main.html')
#     #return render_template('main.html', hourly_count_dict_male=hourly_count_dict_male, hourly_count_dict_female=hourly_count_dict_female, totalMale=totalMale, totalFemale=totalFemale, frame=frame)
    totalMale = 0
    totalFemale = 0
    hourly_count_dict_male = dict((x,0) for x in np.arange(24))
    hourly_count_dict_female = dict((x,0) for x in np.arange(24))
   
    def gen():

    # initialize our centroid tracker and frame dimensions
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
    hourly_count_dict_male = dict((x,0) for x in np.arange(24))
    hourly_count_dict_female = dict((x,0) for x in np.arange(24))

    
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

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    cap =cv2.VideoCapture(0)

    # for IP Camera
    #cap = cv2.VideoCapture('rtsp://admin:admin@172.16.0.14')

    time.sleep(2.0)

    # loop over the frames from the video stream

    while True:
        # read the next frame from the video stream and resize it
        ret,frame = cap.read()
        cv2.imwrite('test.jpg', frame)
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

                blob2 = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)


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
                    if gender is 'Male':
                        totalMale += 1
                        to.counted = True
                        hourly_count_dict_male[now.hour]+=1
                        print('Male:\n',hourly_count_dict_male)

                    elif gender is 'Female':
                        totalFemale += 1
                        to.counted = True
                        hourly_count_dict_female[now.hour]+=1
                        print('Female:\n',hourly_count_dict_female)
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Male", totalMale),
            ("Female", totalFemale),
            ("Total Count", totalMale+totalFemale)
        ]   

        ## loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        #ret, jpeg = cv2.imencode('.jpg', frame)
        #ret, jpeg = cv2.imshow("Frame", frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        socketio.emit('newframe', {'hourly_count_dict_male': hourly_count_dict_male, 'hourly_count_dict_female': hourly_count_dict_female,'totalMale':totalMale, 'totalFemale':totalFemale })

        # ret, jpeg = cv2.imencode('.jpg', frame)
        # newJp = jpeg.tobytes()
        # print(newJp)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("q")
            break

    yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
    cap.release()
    writer.release()
    cv2.destroyAllWindows() 
     
    return jpeg
    


@app.route('/')
def chart1():
    #return render_template('main.html')
    totalMale = 6
    totalFemale = 7
    hourly_count_dict_male = {0: 0, 1: 0, 2: 0, 3: 2, 4: 4, 5: 4, 6: 2, 7: 10, 8: 1, 9: 8, 10: 4, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
    hourly_count_dict_female = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 5, 6: 7, 7: 6, 8: 4, 9: 1, 10: 6, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
    print(totalMale)
    return render_template('main.html', hourly_count_dict_male=hourly_count_dict_male, hourly_count_dict_female=hourly_count_dict_female, totalMale=totalMale, totalFemale=totalFemale)
    socketio.emit('newframe', {'hourly_count_dict_male': hourly_count_dict_male, 'hourly_count_dict_female': hourly_count_dict_female,'totalMale':totalMale, 'totalFemale':totalFemale} , namespace='/test')
    # app.add_url_rule('/chart', 'chart1', chart1)

@app.route('/video_feed')
def video_feed():
    print('video feed is called...')
    genres = gen()
    print(genres)
    return Response(genres,
                    mimetype='image/jpg; boundary=frame')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
  
    print('Client connected')
     
    # #Start the random number generator thread only if the thread has not been started before.
    # if not thread.isAlive():
    #     print("Starting Thread")
    #     thread = gen()
    #     thread.start()

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')
if __name__ == "__main__":
# app.run(debug=True)
    socketio.run(app, debug=True)  


# do a bit of cleanup

