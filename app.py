import os, json
import time
# import eyetraking.main
import env

import redis
from flask import Flask, render_template, redirect, request, jsonify
from eyetraking.main import main, predict, uploadImg
from flask_cors import CORS, cross_origin
from PIL import Image
import pymysql.cursors

app = Flask(__name__, static_folder='eyetraking')
CORS(app)
cache = redis.Redis(host=env.hostCache, port=6379)

# Connect to the database
connection = pymysql.connect(
    host=env.hostDB,
    user=env.userDB,
    password=env.passwordDB,
    db=env.db,
    charset='utf8mb4',
    port = 3306,
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # Create a new record
        sql = "SELECT * FROM `users`"
        cursor.execute(sql)
        result = cursor.fetchone()
        print(result)
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    # connection.commit()
    
finally:
    connection.close()

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

@app.route('/api/')
def eyetraking():
    return '<h1>Ruta base API</h1>'

@app.route('/api/predict')
def predicts():    
    # os.system('python /code/eyetraking/main.py test /code/eyetraking/sample_images/')
    imagenes = predict()
    return jsonify({'response': True, 'data': imagenes }) # respuesta para api
    return redirect('/api/getPrediction') # Retorno de la app funcionando solo en python

@app.route('/api/delete')
def delete():    
    name = request.args.get('name')
    os.remove( '/code/eyetraking/sample_images/{}'.format(name) ) 
    os.remove( '/code/eyetraking/predictions/{}'.format(name) ) 
    return json.dumps({'response': True }) # respuesta para api
    return redirect('/api/getPrediction') # Retorno de la app funcionando solo en python
    return '<h1>The name is: {}</h1>'.format(name)

@app.route("/api/img", methods=["POST"])
def process_image():
    uploadImg()
    file = request.files['imagefile']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img.save('/code/eyetraking/sample_images/{}'.format(file.filename), img.format)    
    return jsonify({'response': True, 'url': 'http://localhost:5000/eyetraking/sample_images/{}'.format(file.filename) }) # respuesta para api
    return redirect('/api/getPrediction') # Retorno de la app funcionando solo en python
    return jsonify({'msg': 'success', 'size': [img.width, img.height]})

@app.route('/api/getPrediction')
def getPrediction():
    # send_from_directory('eyetraking/sample_images', '')
    # return render_template('home.html') 
    return main() 
    return '<h1>eyetraking</h1>'
