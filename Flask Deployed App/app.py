import os
import io
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import boto3

# Load CSVs and model
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# AWS S3 configuration
s3 = boto3.client('s3')
BUCKET_NAME = 'my-plant-images'  # <-- Replace with your actual S3 bucket name

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        # Read image into memory
        image_stream = image.stream.read()

        # Upload to S3
        s3.upload_fileobj(
            io.BytesIO(image_stream),
            BUCKET_NAME,
            filename,
            ExtraArgs={'ACL': 'public-read'}  # Optional: makes file publicly accessible
        )

        # Get S3 image URL
        s3_image_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{filename}"

        # Save locally for model prediction
        TEMP_DIR = 'temp_uploads'
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_path = os.path.join(TEMP_DIR, filename)
        with open(temp_path, 'wb') as f:
            f.write(image_stream)

        # Run prediction
        pred = prediction(temp_path)

        # Optional: remove temp file after prediction
        os.remove(temp_path)

        # Fetch prediction results
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('submit.html',
                               title=title,
                               desc=description,
                               prevent=prevent,
                               image_url=image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image_url,
                               buy_link=supplement_buy_link,
                               uploaded_image_url=s3_image_url)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
