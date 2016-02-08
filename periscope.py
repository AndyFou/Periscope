import os
from os import remove
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import psycopg2
import sys
import logging
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
import argparse
from shutil import copy
# from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif','bmp'])  #TO-DO: check for uppercase image extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

conn = psycopg2.connect("dbname=periscope user=postgres password=morfeaki")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/createdb')
def createdb():
    cur = conn.cursor()
    # cur.execute('DROP TABLE images')
    cur.execute('CREATE TABLE images (id serial PRIMARY KEY, img varchar, color varchar, shape varchar, texture varchar);')
    # cur.execute("INSERT INTO images (img, color, shape, texture) VALUES ('lalalalalal', '{3.0, 11.5, 28.3}', '{3.4}', '{1.2}');")

    conn.commit()
    cur.close()
    # conn.close()          #TO CLOSE AT SOME POINT

    return 'This is an outrage!'

@app.route('/', methods=['GET','POST'])
def addimg():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            #  GET IMAGE
            # filename = secure_filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))        #TO BE ADDED
            copy(os.path.join(app.config['UPLOAD_FOLDER'],file.filename),file.filename)

            #  COMPUTE COLOR HISTOGRAM
            image = cv2.imread(file.filename)               #Read Image
            os.remove(file.filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  #Convert to HSV
            bins = (8, 12, 3)                               #Define the bins that will be extracted
            querycolorfeatures = []

            #Calculate color histogram
            hist = cv2.calcHist([image], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hist = hist.flatten()

            #Convert histogram to color features
            querycolorfeatures.extend(hist)
            querycolorfeatures = [str(f) for f in querycolorfeatures]
            fquerycolorfeatures = [float(x) for x in querycolorfeatures]

            #  SEARCH
            cur = conn.cursor()
            cur.execute("SELECT img, color FROM images;")
            dbimages = cur.fetchall()

            results = {}
            for item in dbimages:
                url = item[0]
                currcolorfeatures = item[1]

                currcolorfeatures = currcolorfeatures.split(",")
                currcolorfeatures[0] = currcolorfeatures[0][1:]
                currcolorfeatures[-1] = currcolorfeatures[-1][:-1]
                currcolorfeatures = [float(x) for x in currcolorfeatures]

                # Calculate ChiSquare distance
                d = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10) for (a, b) in zip(currcolorfeatures, fquerycolorfeatures)])
                results[url] = d

            #  SAVE IMAGE TO DATABASE
            imgurl = url_for('uploaded_file', filename=file.filename);
            cur.execute("INSERT INTO images (img, color) VALUES(%s, %s)", ("'" + imgurl + "'",querycolorfeatures))
            conn.commit()
            cur.close()
            # return redirect(url_for('uploaded_file',filename=file.filename))

            #Get results
            results = sorted([(feature, index) for (index, feature) in results.items()])
            urls = []
            for key, value in results:
                urls.append(value)
            return render_template('results.html', urls=urls[:8])

    return render_template('index.html')

#app.logger.info(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.run(debug=True)
