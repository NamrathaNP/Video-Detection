from flask import Flask, render_template, Response, request, redirect,url_for
# from camera import VideoCamera
import cv2
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from datetime import datetime
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import argparse
import imutils
import time as importedtime
import json
import csv
# import Flask-MySQL
from flaskext.mysql import MySQL
# from flask.ext.mysql import MySQL
import os
import eventlet 


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)
eventlet.monkey_patch() 
socketio = SocketIO(app,async_mode = 'eventlet',  logger=True, engineio_logger=True, ping_timeout=2000, ping_interval=100)

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'root'
# app.config['MYSQL_DB'] = 'vpproj'
mysql = MySQL()
app.config['MYSQL_DATABASE_HOST'] = "127.0.0.1"
app.config['MYSQL_DATABASE_USER'] = 'affine'
app.config['MYSQL_DATABASE_PASSWORD'] = 'affine@123'
app.config['MYSQL_DATABASE_DB'] = 'vproj'
mysql.init_app(app)

# mysql = MySQL(app)

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
gender = None
totalMale = 0
totalFemale = 0
MaleLive = {}
FemaleLive = {}
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

cameraid = "172.16.0.14"
# cameraid = '0'
print("[INFO] starting video stream...")
class VideoCamera(object):
    def __init__(self,cameraid):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        if (len(cameraid)> 2 ):
        
            address= 'rtsp://admin:admin@';
            
            # self.video  = cv2.VideoCapture('rtsp://admin:admin@172.16.0.14')
            self.video  = cv2.VideoCapture(address+cameraid)

        
        else:
        
            cam= int(cameraid)
            self.video = cv2.VideoCapture(cam)
        
        # cam= int(cameraid)
        # self.video = cv2.VideoCapture(cam)
        # # print(type(cam))
        # # print(cam)
        # #video capture from IP webcam
        # address= 'rtsp://admin:admin@';
        # print(cameraid)
        # self.video  = cv2.VideoCapture('rtsp://admin:admin@172.16.0.14')
        # self.video  = cv2.VideoCapture(address+cameraid)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    # def __del__(self):
    #     self.video.release()

    def get_frame(self):
        global cameraid,totalMale, totalFemale, hourly_count_dict_male, hourly_count_dict_female, face_net,gender_net, confidence,gender_list, MODEL_MEAN_VALUES,now, gender
        (H, W) = (None, None)

        writer = None

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
        
        if writer is not None:

            writer.write(frame)

 

            

            if cv2.waitKey(1)&0xFF==ord('q'):
                pass
                # break

        else:
            pass
            # break

       

        if now.hour != datetime.now().hour:

            writer.release()

            print('now hour updated')

            now = datetime.now()

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            writer = cv2.VideoWriter('./video_archive/'+now.strftime('%d-%m-%Y_%H-%M-%S')+'.avi', fourcc,24,(W, H),True)

 

        elif now.minute//15!= datetime.now().minute//15:

            print('now 15min updated')

            writer.release()

            now = datetime.now()

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            writer = cv2.VideoWriter('./video_archive/'+now.strftime('%d-%m-%Y_%H-%M-%S')+'.avi', fourcc,24,(W, H),True)

     
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

                face_img = frame[startY-20:endY+10, startX-20:endX-20].copy()
                if not face_img.size:
                    # print('breaking from face')
                    break

                # blob2 = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)


                # Predict gender
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                # print(gender)
                overlay_text = "%s" % (gender)
                cv2.putText(frame, overlay_text ,(startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)

        cv2.line(frame, (50, 0), (50, H), (0, 255, 255), 1)
        cv2.line(frame, (350, 0), (350, H), (0, 255, 255), 1)          
        # save_frame = frame.copy()       
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
                    print(now)
                    totalFemale=0
                    totalMale=0
                    detecttime = str(now.hour)
                    # print(detecttime)
                    # print(type(detecttime))
                    detectdate = now.strftime('%d-%m-%Y')
                    # print(detectdate)
                    # print(type(detectdate))
                    nowdatetime= now.strftime('%d-%m-%Y_%H-%M-%S')
                    # print(nowdatetime)
                    # print(type(nowdatetime))

                    conn = mysql.connect()
                    cur = conn.cursor()
                    query_string = "SELECT detectHour, sum(maleCount), sum(femaleCount) FROM gcount WHERE updatedTime <= %s AND detectDate = %s GROUP BY detectHour "
                    cur.execute(query_string, (nowdatetime,detectdate))
                    # cur.execute("INSERT INTO gcount(cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (cameraid,detectdate,detecttime,1,0,nowdatetime)")
                    data = list(cur.fetchall())
                    # print(data)
                    cur.close() 

                    for i in data:
                     
                        hourly_count_dict_male[i[0]]=int(i[1])
                        # print(hourly_count_dict_male);
                        hourly_count_dict_female[i[0]]=int(i[2])
                        # print(hourly_count_dict_female);
                    for key in hourly_count_dict_female:
                        totalFemale+=  hourly_count_dict_female[key]
                    for key in hourly_count_dict_male:
                        totalMale+=  hourly_count_dict_male[key]
                        # print(totalFemale)
                    if gender is 'Male':
                        totalMale += 1
                        to.counted = True
                        
                        hourly_count_dict_male[detecttime]+=1
                        
                        print('Male:\n',hourly_count_dict_male)
                       
                        (pd.DataFrame.from_dict(data=hourly_count_dict_male, 
                        orient='index').to_csv('./static/male.csv', header=True))
                        
                        # print("b4 sleep")
                        # hcdm= json.dumps(hourly_count_dict_male)
                        # tM= json.dumps(totalMale)
                        # MaleLive= hourly_count_dict_male
                        conn = mysql.connect()
                        cur = conn.cursor()
                        mySql_insert_query = """INSERT INTO gcount (cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (%s, %s, %s, 1, 0, %s) """
                        recordTuple = (cameraid,detectdate,detecttime,nowdatetime)
                        cur.execute(mySql_insert_query, recordTuple)
                        conn.commit()
                        cur.close()
                       
                        sendingvarmale= json.dumps({'hourly_count_dict_male': hourly_count_dict_male,'totalMale':totalMale})
                       
                        # socketio.emit('newframe', {'display':"Hi" })
                        socketio.emit('newmale', {'resp': sendingvarmale} , namespace='/test')
                        
                        socketio.sleep(1)
                        # print("after sleep")
                       
                        

                    elif gender is 'Female':
                        totalFemale += 1
                        to.counted = True
                       
                        hourly_count_dict_female[detecttime]+=1

                        print('Female:\n',hourly_count_dict_female)
                       
                        # (pd.DataFrame.from_dict(data=hourly_count_dict_female, 
                        # orient='index').to_csv('./static/female.csv', header=True))
                       
                        # print("b4 sleep")
                        # hcdf= hourly_count_dict_female
                        # tF= totalFemale
                        # FemaleLive = hourly_count_dict_female
                        conn = mysql.connect()
                        cur = conn.cursor()
                        mySql_insert_query = """INSERT INTO gcount (cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (%s, %s, %s, 0, 1, %s) """
                        recordTuple = (cameraid,detectdate,detecttime,nowdatetime)
                        cur.execute(mySql_insert_query, recordTuple)

                        # cur.execute("INSERT INTO gcount(cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (cameraid,detectdate,detecttime,0,1,nowdatetime)")
                        conn.commit()
                        cur.close() 
                       
                        sendingvar= json.dumps({'hourly_count_dict_female': hourly_count_dict_female,'totalFemale':totalFemale})
                        # socketio.emit('newframe', {'display':"Hi" })

                        socketio.emit('newfemale', {'resp':sendingvar } , namespace='/test')
                        socketio.sleep(1)
                        # print("after sleep")
                    
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
        # jpeg_sa_text= base64.b64encode('.jpg', frame)
        # return jpeg_sa_text
        return jpeg.tobytes()

# def detect():
#     while True:
#         gen(camera);

@app.route('/')
def index():
    hourly_count_dict_male = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0}
    hourly_count_dict_female = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0}

    sendingvarmale={}
    sendingvar={}
    now = datetime.now()
    detectdate = now.strftime('%d-%m-%Y')
    nowdatetime= now.strftime('%d-%m-%Y_%H-%M-%S')
    totalFemale=0
    totalMale=0
    print("initial_video")
    conn = mysql.connect()
    cur = conn.cursor()
    query_string = "SELECT detectHour, sum(maleCount), sum(femaleCount) FROM gcount WHERE updatedTime <= %s AND detectDate = %s GROUP BY detectHour "
    cur.execute(query_string, (nowdatetime,detectdate))
    # cur.execute("INSERT INTO gcount(cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (cameraid,detectdate,detecttime,1,0,nowdatetime)")
    data = list(cur.fetchall())
    print(data)
    cur.close() 

    for i in data:
     
        hourly_count_dict_male[i[0]]=int(i[1])
        # print(hourly_count_dict_male);
        hourly_count_dict_female[i[0]]=int(i[2])
        # print(hourly_count_dict_female);
    for key in hourly_count_dict_female:
        totalFemale+=  hourly_count_dict_female[key]
    for key in hourly_count_dict_male:
        totalMale+=  hourly_count_dict_male[key]

    
    # print(hourly_count_dict_male)
    # print(totalMale)
    # print(hourly_count_dict_female)
    # print(totalFemale)  
    sendinginidata= ({'hourly_count_dict_male': hourly_count_dict_male,'totalMale':totalMale,'hourly_count_dict_female': hourly_count_dict_female,'totalFemale':totalFemale})
     
    return render_template('main.html',sendinginidata=sendinginidata )
   

def gen(camera):
    global ct, trackers,trackableObjects, writer, now, MODEL_MEAN_VALUES,gender_list,totalMale, totalFemale, hourly_count_dict_male, hourly_count_dict_female,confidence, face_net, gender_net

    print("[INFO] starting video stream...")
   
    while True:
        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
@app.route('/video_feed/<cameraid>')
def video_feed(cameraid):
    print(cameraid)
    print(type(cameraid))
    
    return Response(gen(VideoCamera(cameraid)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/record_data', methods=["GET", "POST"])
def recordlookup():
    hourly_count_dict_female={}
    hourly_count_dict_male={}
    sendingvarmale={}
    sendingvar={}
    cameraid = '172.16.0.14'
    # seltime= 11
    # seldate= '02-12-2019'
    totalFemale=0
    totalMale=0
    # cameraid1 = request.form.get('cameraid')
    seltime = request.form.get('seltime')
    seldate = request.form.get('seldate')
    # print(cameraid1, seldate1, seltime1)

    conn = mysql.connect()
    cur = conn.cursor()
    query_string = "SELECT cameraid, detectDate, detectHour, sum(maleCount), sum(femaleCount) FROM gcount WHERE detectDate = %s  AND detectHour = %s AND cameraId = %s GROUP BY detectHour, cameraid, detectDate"
    cur.execute(query_string, (seldate, seltime, cameraid))
    # cur.execute("INSERT INTO gcount(cameraId,detectDate,detectHour,maleCount,femaleCount,updatedTime) VALUES (cameraid,detectdate,detecttime,1,0,nowdatetime)")
    data = list(cur.fetchall())
    # print(data)
    cur.close() 

    for i in data:
        hourly_count_dict_male[i[0]]=int(i[1])
        # print(hourly_count_dict_male);
        hourly_count_dict_female[i[0]]=int(i[2])
        # print(hourly_count_dict_female);
    for key in hourly_count_dict_female:
        totalFemale+=  hourly_count_dict_female[key]
    for key in hourly_count_dict_male:
        totalMale+=  hourly_count_dict_male[key]

    sendingvarmale= json.dumps({'hourly_count_dict_male': hourly_count_dict_male,'totalMale':totalMale})
                       
                        # socketio.emit('newframe', {'display':"Hi" })
    socketio.emit('newmale', {'resp': sendingvarmale} , namespace='/test')
    sendingvar= json.dumps({'hourly_count_dict_female': hourly_count_dict_female,'totalFemale':totalFemale})
                        # socketio.emit('newframe', {'display':"Hi" })

    socketio.emit('newfemale', {'resp':sendingvar } , namespace='/test')

    return ""




@socketio.on('connect', namespace='/test')
def test_connect():
    
    # need visibility of the global thread object
    print('Client connected')

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    # app.run( debug=True)
    socketio.run(app,host='0.0.0.0', debug=True)

video_feed('172.16.0.14')