from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load the TensorFlow Lite model
model_path = "tflite_learn_3.tflite"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)  # Exit if the model is not found

model = tf.lite.Interpreter(model_path=model_path)
model.allocate_tensors()

# Get model input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

# Labels for prediction
labels = ["pants", "shirt", "shoes", "shorts", "tshirt"]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    try:
        if request.method == 'POST':
            # Get the uploaded image file
            image = request.files['file']
            if image:
                # Open the image, convert it to grayscale and resize to 96x96
                img = Image.open(image).convert("L").resize((96, 96))  # Resize to 96x96 and convert to grayscale

                # Convert image to numpy array and normalize it
                input_data = np.asarray(img).astype(np.float32) / 255.0

                # Reshape the input to match the model's expected input shape (1, 96, 96, 1)
                input_data = input_data.reshape((1, 96, 96, 1))

                # Set the input tensor and run the inference
                model.set_tensor(input_details[0]['index'], input_data)
                model.invoke()

                # Get the output tensor and make a prediction
                output = model.get_tensor(output_details[0]['index'])
                predicted_index = int(np.argmax(output))
                confidence = float(output[0][predicted_index])

                # Format the prediction result
                prediction = {
                    "label": labels[predicted_index],
                    "confidence": round(confidence * 100, 2)
                }
    except Exception as e:
        print(f"Error: {e}")
        prediction = {"label": "Error", "confidence": 0.0}

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
