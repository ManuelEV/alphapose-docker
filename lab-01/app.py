import os
from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from demo2 import DemoInference

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = 'examples/input'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = '5f352379324c22463451387a0aec5d2f'

demo_inference = DemoInference()
demo_inference.load()


demo_inference.process_image()


@app.route('/')
def hello_world():  # put application's code here

    return 'Hello World!'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_img():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print('Uploading file ' + filename + ' as image.jpg')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
        demo_inference.process_image()
        return redirect(url_for('uploaded_file',
                                filename=filename))
    return 'Hello World!'


@app.route('/uploaded')
def uploaded_file():
    return 'Uploaded!'


if __name__ == '__main__':
    app.run()
