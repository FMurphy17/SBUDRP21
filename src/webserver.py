#https://pythonbasics.org/webserver/

import os
import sys
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

hostName = "localhost"
serverPort = 8080

def identify_image(fn):
    image = keras.preprocessing.image.load_img(fn, color_mode="grayscale")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = np.abs(input_arr - 255.0)
    p1 = model.predict(input_arr)

    maxpred = p1.max()
    predindex = int(p1.argmax(axis=-1))
    predlabel = labels[predindex]
    return f"'{predlabel}' with probabilty: {maxpred:.3f}"

class MyServer(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(content_length)
        print(post_data)
        a = str(post_data).split(',')[1]
        png = base64.b64decode(a)
        f = open('x.jpg', 'wb')
        f.write(png)
        f.close()
        label = identify_image("x.jpg")
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-length", len(label))
        self.end_headers()
        self.wfile.write(bytes(label, "utf-8"))


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    labels = sorted([os.path.basename(i) for i in glob.glob("../data/extracted_images/*")])
    model = keras.models.load_model('./testmodel.data')

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")