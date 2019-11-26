from flask import Flask ,request
from flask import render_template
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
# 

from flask_socketio import SocketIO
from random import randint
from time import sleep
from threading import Thread, Event
import os
# from flask_mysqldb import MYSQL
# from flask_cors import CORS
#from datetime import time
#from scipy.io import wavfile # scipy library to read wav files
#import numpy as np
PEOPLE_FOLDER = os.path.join('static')

 
lables=[]
data=[]
data1=[]
data2=[]

# values={'00':['4','5'],'01':['5','7'],'02':['6','8'],'03':['8','7'],'04':['1','3'],'05':['3','7'],'06':['1','4'],'07':['1','2'],'08':['5','9'],'09':['2','4'],'10':['6','8']};
# values={00:[4,5],01:[5,7],02:[6,8],03:[8,7],04:[5,3],05:[3,7],06:[1,4],07:[1,2],8:[5,9],9:[2,4],10:[6,8]};

# for key, value in values.items():
# 	lables.append(key);
# 	data.append(value);
# print(lables)
# print(data)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

thread = Thread()
thread_stop_event = Event()

class RandomThread(Thread):
    def __init__(self):
        self.delay = 2
        super(RandomThread, self).__init__()

    def randomNumberGenerator(self):
        """
        Generate a random number every 1 second and emit to a socketio instance (broadcast)
        Ideally to be run in a separate thread?
        """
        #infinite loop of magical random numbers
        print("Making random numbers")
        k = 0
        while not thread_stop_event.isSet():
            
            for k in range(10):
            # while(k<=10):
                
                data2.append(k) 
                print({ "range k": k, "data" : data2})

                k = k+1
                i = randint(1,10)
                j = randint(1,10)
                b= [i,j]
                    
                data1.append(b)
                socketio.emit('my_msg',{'k': k} , namespace='/test')

            print(data1)
            print(data2)
            
            socketio.emit('newnumber', {'data1': data1, 'data2': data2,'display':"HI"} , namespace='/test')
            socketio.send("Hi")
            sleep(self.delay)
      

    def run(self):
        
      self.randomNumberGenerator()
    
    
            
# api = Api(app) 
# app.config['MYSQL_USER']= 'root@localhost';
# app.config['MYSQL_PASSWORD']= '';
# app.config['MYSQL_DB']= 'vproj';
# app.config['MYSQL_CURSORCLASS']= 'DictCursor';
# mysql=MySQL(app);

@app.route("/")
def chart():
    image=os.path.join(app.config['UPLOAD_FOLDER'], 'download.jpg') 
	# conn = mysql.connect()
	# cursor =conn.cursor()

	# cur.execute("SELECT * FROM empdata")
	# res= cur.fetchall();
	# return jsonify(res);
    return render_template('demo-orig.html', lables=lables, data=data, image=image)
 
# def messageReceived(methods=['GET', 'POST']):
#     print('message was received!!!')

# @socketio.on('my event')
# def handle_my_custom_event(json, methods=['GET', 'POST']):
#     print('received my event: ' + str(json))
#     socketio.emit('my response', json, callback=messageReceived)

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = RandomThread()
        thread.start()

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

 
if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, debug=True)