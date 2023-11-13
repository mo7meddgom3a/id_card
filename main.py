from id import read_id_front, read_id_back
from flask import Flask, render_template, request, url_for
import os
import cv2


app = Flask(__name__)
# Define the folder where uploaded images will be stored
UPLOAD_FOLDER_PATH = "./uploaded_images"

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER_PATH):
    os.makedirs(UPLOAD_FOLDER_PATH)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/uploadfront", methods=["POST"])
def upload_image():
    if "image" in request.files:
        image = request.files["image"]
        if image.filename != "":
            img_path = os.path.join(UPLOAD_FOLDER_PATH, 'front_id',image.filename)
            image.save(img_path)
            osrResult = read_id_front(cv2.imread(img_path))
            print(osrResult)
            return {"id":osrResult}

@app.route("/uploadback", methods=["POST"])
def upload_image_back():
    if "image" in request.files:
        image = request.files["image"]
        if image.filename != "":
           img_path = os.path.join(UPLOAD_FOLDER_PATH, 'back_id',image.filename)
           image.save(img_path)
           ocrResult = read_id_back(cv2.imread(img_path))
           print(ocrResult)
    return {"id":ocrResult}

if __name__ == "__main__":
    app.run(port=5000, debug=True)
    print(f"Server is running at port 5000")