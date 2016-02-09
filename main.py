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
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

app = Flask(__name__)

#APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif','bmp'])  #TO-DO: check for uppercase image extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/createdb')
def createdb():
    conn = psycopg2.connect("dbname=root user=root password=qwe")
    cur = conn.cursor()

    cur.execute('CREATE TABLE images (id serial PRIMARY KEY, img varchar, color varchar, shape varchar, texture varchar);')
    conn.commit()

    cur.close()
    conn.close()

    return 'Table created!'

@app.route('/', methods=['GET','POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            conn = psycopg2.connect("dbname=root user=root password=qwe")

            #  GET IMAGE
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))        #TO BE ADDED
            copy(os.path.join(app.config['UPLOAD_FOLDER'],file.filename),file.filename)

            #  COMPUTE COLOR
            image = cv2.imread(file.filename)                    #Read Image
            os.remove(file.filename)

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   #Convert to HSV
            bins = (8, 12, 3)                                    #Define the bins that will be extracted
            querycolorfeatures = []
            querytexturefeatures = []

            #Calculate color histogram
            color_hist = cv2.calcHist([image_hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])
            cv2.normalize(color_hist, color_hist)
            color_hist = color_hist.flatten()

            #Convert histogram to color features
            querycolorfeatures.extend(color_hist)
            querycolorfeatures = [str(f) for f in querycolorfeatures]
            fquerycolorfeatures = [float(x) for x in querycolorfeatures]

            #  COMPUTE TEXTURE

            #Calculate texture histogram
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Convert to HSV
            radius = 3
            no_points = 8 * radius                                # Number of points to be considered as neighbourers
            lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
            x = itemfreq(lbp.ravel())
            texture_hist = x[:, 1]/sum(x[:, 1])

            #Convert histogram to color features
            querytexturefeatures.extend(texture_hist)
            fquerytexturefeatures = [float(x) for x in querytexturefeatures]

            #  PERFORM SEARCH
            querytotalfeature = fquerycolorfeatures + fquerytexturefeatures

            cur = conn.cursor()
            cur.execute("SELECT img, color, texture FROM images;")
            dbimages = cur.fetchall()

            results = {}
            for item in dbimages:
                url = item[0]
                currcolorfeatures = item[1]
                currtexturefeatures = item[2]

                # Get db-images color features
                currcolorfeatures = currcolorfeatures.split(",")
                currcolorfeatures[0] = currcolorfeatures[0][1:]
                currcolorfeatures[-1] = currcolorfeatures[-1][:-1]
                currcolorfeatures = [float(x) for x in currcolorfeatures]

                # Get db-images texture features
                currtexturefeatures = currtexturefeatures.split(",")
                currtexturefeatures[0] = currtexturefeatures[0][1:]
                currtexturefeatures[-1] = currtexturefeatures[-1][:-1]
                currtexturefeatures = [float(x) for x in currtexturefeatures]

                currtotalfeature = currcolorfeatures + currtexturefeatures

                # Calculate ChiSquare distance
                # d = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10) for (a, b) in zip(currcolorfeatures, fquerycolorfeatures)])
                d = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10) for (a, b) in zip(currtotalfeature, querytotalfeature)])
                results[url] = d

            #  SAVE IMAGE TO DATABASE
            imgurl = url_for('uploaded_file', filename=file.filename);
            cur.execute("INSERT INTO images (img, color, texture) VALUES(%s, %s, %s)", ("'" + imgurl + "'",querycolorfeatures,querytexturefeatures))
            conn.commit()
            cur.close()
            conn.close()
            # return redirect(url_for('uploaded_file',filename=file.filename))

            #  GET SEARCH RESULTS
            results = sorted([(feature, index) for (index, feature) in results.items()])
            urls = []
            for key, value in results:
                urls.append(value)
            return render_template('results.html', urls=urls[:8])

    return render_template('index.html')

@app.route('/addimg', methods=['GET','POST'])
def addimg():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            conn = psycopg2.connect("dbname=root user=root password=qwe")

            #  GET IMAGE
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))        #TO BE ADDED
            copy(os.path.join(app.config['UPLOAD_FOLDER'],file.filename),file.filename)

            #  COMPUTE COLOR
            image = cv2.imread(file.filename)                    #Read Image
            os.remove(file.filename)

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   #Convert to HSV
            bins = (8, 12, 3)                                    #Define the bins that will be extracted
            querycolorfeatures = []
            querytexturefeatures = []

            #Calculate color histogram
            color_hist = cv2.calcHist([image_hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])
            cv2.normalize(color_hist, color_hist)
            color_hist = color_hist.flatten()
            querycolorfeatures.extend(color_hist)
            querycolorfeatures = [str(f) for f in querycolorfeatures]

            #  COMPUTE TEXTURE

            #Calculate texture histogram
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Convert to HSV
            radius = 3
            no_points = 8 * radius                                # Number of points to be considered as neighbourers
            lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
            x = itemfreq(lbp.ravel())
            texture_hist = x[:, 1]/sum(x[:, 1])
            querytexturefeatures.extend(texture_hist)

            #  SAVE IMAGE TO DATABASE
            imgurl = url_for('uploaded_file', filename=file.filename);
            cur = conn.cursor()
            cur.execute("INSERT INTO images (img, color, texture) VALUES(%s, %s, %s)", ("'" + imgurl + "'",querycolorfeatures,querytexturefeatures))
            conn.commit()
            cur.close()
            conn.close()
            # return redirect(url_for('uploaded_file',filename=file.filename))

    return render_template('addimg.html')

#app.logger.info(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
