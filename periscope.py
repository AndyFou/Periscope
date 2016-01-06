import os
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import psycopg2

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
    cur.execute('CREATE TABLE images (id serial PRIMARY KEY, img varchar, color real[], shape real[], texture real[]);')
    cur.execute("INSERT INTO images (img, color, shape, texture) VALUES ('lalalalalal', '{3.0, 11.5, 28.3}', '{3.4}', '{1.2}');")

    conn.commit()
    cur.close()
    conn.close()

    return 'This is an outrage!'

@app.route('/addimg', methods=['GET','POST'])
def addimg():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # filename = secure_filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            return redirect(url_for('uploaded_file',filename=file.filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
app.run(debug=True)
