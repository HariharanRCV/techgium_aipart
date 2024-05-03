from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import time
import pyzbar.pyzbar as pyzbar
import base64
from flask_socketio import emit
import csv
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

model_config_path = r'C:\Users\harih\L&T_Techgium\techgium_aipart (all functions updated with backend)\techgium_aipart\tech\final_best1.h5'  # Replace with the actual path
model = load_model(model_config_path)

rack_results = {i: None for i in range(1, 21)}  # Initialize results for each rack
rack_number = 1  # Initialize rack number

def resize_image(image, size=(224, 224)):
    resized_img = cv2.resize(image, size)
    return resized_img

def preprocess_input_image(image):
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet.preprocess_input(image_array)
    return image_array

def predict_image_segmentation(model, image):
    input_image = preprocess_input_image(image)
    prediction = model.predict(input_image)
    binary_mask = prediction > 0.5
    return binary_mask[0, :, :, 0]

def run_qr_scanner(frame):
    if frame is None:
        print("Error: Couldn't load the image.")
        qr_output = None 
    else:
        decoded_objects = pyzbar.decode(frame)
        qr_output = []
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            qr_output.append(qr_data)   
    return qr_output

rack_number = 0

####################################################

# Initialize a list to accumulate data for each rack
accumulated_data = []

def generate_folder_path():
    current_datetime = datetime.now()
    folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("csv_backups", folder_name)  # Replace "path_to_your_folder" with the desired folder path
    return folder_path

def save_to_csv(folder_path):
    global accumulated_data

    # Check if accumulated_data contains data for at least 20 racks
    if len(accumulated_data) >= 20:
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define the CSV file name
        csv_file_name = 'rack_data.csv'

        # Write accumulated data to a CSV file in the specified folder
        csv_file_path = os.path.join(folder_path, csv_file_name)
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['Rack Number', 'Prediction Result', 'QR Result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in accumulated_data:
                writer.writerow(data)

        # Reset accumulated data for the next set of racks
        accumulated_data = []

####################################################

# ...
captured_frame_base64 = ""
prediction_running = False

def capture_and_process_image(camera_index=0):
    global rack_results
    global rack_number
    global captured_frame_base64##########
    global accumulated_data##########
    global prediction_running
    
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    
    _, buffer = cv2.imencode('.jpg', frame)##########
    captured_frame_base64 = base64.b64encode(buffer).decode('utf-8')###########


    resized_image = resize_image(frame)

    predicted_mask = predict_image_segmentation(model, resized_image)

    white_pixel_count = np.sum(predicted_mask)

    prediction_result = "1" if white_pixel_count > 2000 else "0"
    print(f"Rack {rack_number} : {prediction_result}")

    qr_result = None  # Initialize qr_result

    if prediction_result == "1":
        qr_result = run_qr_scanner(frame)
        if not qr_result:
            print(f"Rack {rack_number} - No QR found or unrecognizable")
        else:
            print(f"Rack {rack_number} - QR Result: {qr_result}")
    elif prediction_result == "0":
        print(f"Rack {rack_number} -  product not available")

    # Move to the next rack
    rack_number = (rack_number % 20) + 1
    
    rack_results[rack_number] = {'prediction_result': prediction_result, 'qr_result': qr_result}

    socketio.emit('update_result', {'rack_results': rack_results})
    socketio.emit('update_image', {'captured_frame': captured_frame_base64})

    accumulated_data.append({'Rack Number': rack_number, 'Prediction Result': prediction_result, 'QR Result': qr_result})#########

    if rack_number % 20 == 0:
    # Generate folder path based on current date and time
        folder_path = generate_folder_path()

        # Save accumulated data to CSV file in the specified folder
        save_to_csv(folder_path)
        prediction_running = False
    # ...



@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(target=continuous_processing)


####################
@socketio.on('start_prediction')
def start_prediction():
    global prediction_running
    prediction_running = True

@socketio.on('stop_prediction')
def stop_prediction():
    global prediction_running
    prediction_running = False
####################


def continuous_processing(camera_index=0):
    global rack_results
    global rack_number
    global prediction_running

    while True:
        if prediction_running:
            capture_and_process_image(camera_index)
        time.sleep(7)  # Wait for 5 seconds between scans

# Placeholder function to fetch billing data based on rack number
def fetch_billing_data(rack_number):
    for data in accumulated_data:
        if data['Rack Number'] == rack_number:
            return data
    return None

# Flask route to handle fetching billing information
# Flask route to handle fetching billing information
@socketio.on('fetch_billing')
def fetch_billing(data):
    rack_number = int(data['rackNumber'])
    customer_name = data['customerName']
    rack_price = data['rackPrice']
    product_quantity = data['productQuantity']
    customer_mobile = data['customerMobile']
    
    billing_data = fetch_billing_data(rack_number)
    if billing_data:
        # Emit billing data, QR result, customer name, rack price, product quantity, and customer mobile number
        socketio.emit('update_billing', {
            'billingResult': billing_data,
            'qrResult': billing_data['QR Result'],
            'customerName': customer_name,
            'rackPrice': rack_price,
            'productQuantity': product_quantity,
            'customerMobile': customer_mobile
        })
    else:
        socketio.emit('update_billing', {'billingResult': None, 'qrResult': None})


@app.route('/')
def home():
    global captured_frame_base64
    #return render_template('index.html', rack_results=rack_results)
    return render_template('index.html', rack_results=rack_results, captured_frame=captured_frame_base64)

if __name__ == '__main__':
    socketio.run(app, debug=True)
