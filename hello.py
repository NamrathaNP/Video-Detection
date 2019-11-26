from flask import Flask
from flask import render_template
from datetime import time
from scipy.io import wavfile # scipy library to read wav files

import numpy as np

 

AudioName = "./load1800r1-input1( 0.00-60.00 s).wav" # Audio File

fs, Audiodata = wavfile.read(AudioName)

t = np.arange(len(Audiodata)) / float(fs)
values=[]
for i in Audiodata:
	#print (i);
	values.append(i);

#print(values)
print(Audiodata)
app = Flask(__name__)
 
 
@app.route("/")
def chart():

    return render_template('hello.html', values=values)
 
 
if __name__ == "__main__":
    app.run(debug=True)