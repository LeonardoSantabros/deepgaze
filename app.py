import os
import time
# import eyetraking.main
from eyetraking.main import main, predict

import redis
from flask import Flask, render_template, redirect

app = Flask(__name__, static_folder='eyetraking')
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count) 

@app.route('/about')
def about():
    return '<h1>About</h1>'

@app.route('/api/predict')
def predicts():    
    # os.system('python /code/eyetraking/main.py test /code/eyetraking/sample_images/')
    predict()
    return redirect('/api/getPrediction')

@app.route('/api/getPrediction')
def getPrediction():
    # send_from_directory('eyetraking/sample_images', '')
    # return render_template('home.html') 
    return main() 
    return '<h1>eyetraking</h1>'
