from flask import Flask, render_template, request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from src.extract_math_eq import *
import base64
import uuid
import json

import os

app = Flask(__name__)

new_dir = "tmp_file/"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/api/classify', methods=['POST'])
def upload_image():
    img = None
    print("new request!!!")
    if request.is_json:
        data = request.get_json()
        if type(data) is str:
            data = json.loads(data)
        img_encode = data["img_encode"]
        img_format = data["format"]

        if img_format == "file":
            file_path = "tmp_file/" + str(uuid.uuid4())
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(img_encode))

            img = cv2.imread(file_path)
            os.remove(file_path)
        elif img_format == "opencv":
            print("unsupport")
    else:
        f = request.files['image']
        file_path = "tmp_file/" + str(uuid.uuid4())
        f.save(file_path)

        img = cv2.imread(file_path)

        os.remove(file_path)

    eq_string, _ = extract_mat_eq(img)

    return eq_string


def dev_server():
    app.run()


def server():
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(80)
    IOLoop.instance().start()


if __name__ == '__main__':
    dev_server()
