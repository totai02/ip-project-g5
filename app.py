from flask import Flask, render_template, request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from src.extract_math_eq import *
import base64
import random
import json

import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/api/classify', methods=['POST'])
def upload_image():
    eq_string = ""
    img = None

    if request.is_json:
        content = request.get_json()
        data = json.loads(content)
        img_encode = data["img_encode"]
        img_format = data["format"]

        if img_format == "file":
            tmp_name = str(random.randint(0, 10000)) + ".jpg"
            file_path = "tmp_file/" + tmp_name
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(img_encode))

            img = cv2.imread(file_path)
            os.remove(file_path)
        elif img_format == "opencv":
            print("unsupport")

    eq_string = extract_mat_eq(img)

    return eq_string


def dev_server():
    app.run(debug=True)


def server():
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)
    IOLoop.instance().start()


if __name__ == '__main__':
    dev_server()
