# from flask import Flask, request, jsonify
# import pandas as pd
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)

# Load the TensorFlow model
# model = tf.keras.models.load_model("improved_recliner_model.h5")


# @app.route("/predict", methods=["POST"])
# def predict():
# try:
# Parse input data from JSON
# data = request.json

# Prepare input data (ensure it matches the model's expected input shape)
# input_data = pd.DataFrame({
# "Height": [data["Height"]],
# "Weight": [data["Weight"]],
# "WorkSittingTime": [data["WorkSittingTime"]],
# "Gender_Male": [data["Gender_Male"]],
# "CommuteType_Car": [data["CommuteType_Car"]],
# "CommuteType_Train": [data["CommuteType_Train"]]
# Exclude "CommuteType_Walk" to match the expected input shape of 7 features
# })

# Add BMI as a derived feature (if required by the model)
# input_data["BMI"] = input_data["Weight"] / (input_data["Height"] ** 2)

# Ensure the order of columns matches the model's training data
# input_data = input_data[[
# "Height",
# "Weight",
# "BMI",
# "WorkSittingTime",
# "Gender_Male",
# "CommuteType_Car",
# "CommuteType_Train"
# ]]

# Convert DataFrame to NumPy array for prediction
# input_array = np.array(input_data)

# Make prediction
# prediction = model.predict(input_array)

# Return result as JSON
# return jsonify({"prediction": prediction.tolist()})

# except Exception as e:
# Return error message if something goes wrong
# return jsonify({"error": str(e)}), 400


# if __name__ == "__main__":
# app.run(debug=True)


# improved_recliner_model
# improved_scaler


# import os

# from flask import Flask

# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     """Example Hello World route."""
#     name = os.environ.get("NAME", "World")
#     return f"Hello {name}!"


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# ++++++++++++++++++++++++++++++  h5 model ++++++++++++++++++++++++++++++

# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import tensorflow as tf
# import joblib
# import os
# import requests

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # os.environ["TF_METAL_DISABLE"] = "1"
# # tf.config.set_visible_devices([], 'GPU')

# app = Flask(__name__)

# # Load model and scaler at startup
# model = None
# scaler = None

# # Firebase Storage URLs (Replace with actual URLs)
# # model_url = "https://firebasestorage.googleapis.com/v0/b/dot-chair.appspot.com/o/models%2Fimproved_recliner_model.h5?alt=media&token=0d6f344b-f021-4f94-896f-b35d178e92d0"
# # scaler_url = "https://firebasestorage.googleapis.com/v0/b/dot-chair.appspot.com/o/models%2Fimproved_scaler.pkl?alt=media&token=dea86162-e9c4-4839-9c2e-91423fa028e4"


# # def download_file(url, save_path):
# #     """Downloads a file from a given URL and saves it locally."""
# #     response = requests.get(url)
# #     if response.status_code == 200:
# #         with open(save_path, "wb") as f:
# #             f.write(response.content)
# #         print(f"Downloaded {save_path}")
# #     else:
# #         print(f"Failed to download {save_path}")


# def load_model_and_scaler():
#     """Loads model and scaler from Firebase Storage"""
#     # print("Downloading and loading model...")

#     # Define file paths
#     model_path = "improved_recliner_model.h5"
#     scaler_path = "improved_scaler.pkl"

#     # Download files
#     # download_file(model_url, model_path)
#     # download_file(scaler_url, scaler_path)

#     # Load model and scaler
#     global model, scaler
#     model = tf.keras.models.load_model(model_path)
#     scaler = joblib.load(scaler_path)

#     print("Model and scaler loaded successfully!")


# # Ensure the model and scaler are loaded when the app starts
# load_model_and_scaler()


# def preprocess_input(input_data):
#     """Preprocess user input to match model requirements"""
#     # Create DataFrame from input
#     df = pd.DataFrame(input_data, index=[0])

#     # Calculate BMI if not provided
#     if 'BMI' not in df.columns or pd.isna(df['BMI'][0]):
#         df['BMI'] = df['Weight'] / (df['Height'] ** 2)

#     # Ensure all expected columns are present
#     expected_columns = [
#         'Height', 'Weight', 'BMI', 'WorkSittingTime',
#         'Gender_Male', 'Gender_Female',
#         'CommuteType_Car', 'CommuteType_Train', 'CommuteType_Walk'
#     ]

#     # Add missing columns with default values
#     for col in expected_columns:
#         if col not in df.columns:
#             if col.startswith('Gender_'):
#                 # Default to male if not specified
#                 df['Gender_Male'] = 1
#                 df['Gender_Female'] = 0
#             elif col.startswith('CommuteType_'):
#                 # Default to car if not specified
#                 df['CommuteType_Car'] = 1
#                 df['CommuteType_Train'] = 0
#                 df['CommuteType_Walk'] = 0
#             else:
#                 df[col] = 0

#     # Add interaction terms
#     df['Height_Weight_Interaction'] = df['Height'] * df['Weight']
#     df['BMI_SittingTime_Interaction'] = df['BMI'] * df['WorkSittingTime']

#     # Reorder columns to match training data
#     df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

#     # Scale features
#     scaled_data = scaler.transform(df)

#     return scaled_data


# @app.route('/', methods=['GET'])
# def home():
#     """Render the home page with input form"""
#     return render_template('index.html')


# @app.route('/health')
# def health_check():
#     return {"status": "healthy"}, 200


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle prediction requests"""
#     try:
#         # Get input data from form
#         input_data = {
#             'Height': float(request.form['height']),
#             'Weight': float(request.form['weight']),
#             'WorkSittingTime': float(request.form['sitting_time']),
#             'Gender_Male': 1 if request.form['gender'] == 'male' else 0,
#             'Gender_Female': 1 if request.form['gender'] == 'female' else 0,
#             'CommuteType_Car': 1 if request.form['commute_type'] == 'car' else 0,
#             'CommuteType_Train': 1 if request.form['commute_type'] == 'train' else 0,
#             'CommuteType_Walk': 1 if request.form['commute_type'] == 'walk' else 0
#         }

#         # Preprocess input
#         processed_input = preprocess_input(input_data)

#         # Make prediction
#         prediction = model.predict(processed_input)
#         predicted_angle = float(prediction[0][0])

#         # Return result
#         return render_template('index.html',
#                                prediction_text=f'Recommended recliner angle: {predicted_angle:.1f}°',
#                                show_result=True)

#     except Exception as e:
#         return render_template('index.html',
#                                prediction_text=f'Error: {str(e)}',
#                                show_result=True)


# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """API endpoint for predictions"""
#     try:
#         # Get JSON data
#         data = request.get_json()

#         # Validate required fields
#         required_fields = ['height', 'weight', 'work_sitting_time']
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({'error': f'Missing required field: {field}'}), 400

#         # Prepare input data
#         input_data = {
#             'Height': float(data['height']),
#             'Weight': float(data['weight']),
#             'WorkSittingTime': float(data['work_sitting_time']),
#             'Gender_Male': 1 if data.get('gender', 'male').lower() == 'male' else 0,
#             'Gender_Female': 1 if data.get('gender', 'male').lower() == 'female' else 0,
#             'CommuteType_Car': 1 if data.get('commute_type', 'car').lower() == 'car' else 0,
#             'CommuteType_Train': 1 if data.get('commute_type', 'car').lower() == 'train' else 0,
#             'CommuteType_Walk': 1 if data.get('commute_type', 'car').lower() == 'walk' else 0
#         }

#         # Preprocess input
#         processed_input = preprocess_input(input_data)

#         # Make prediction
#         prediction = model.predict(processed_input)
#         predicted_angle = float(prediction[0][0])

#         # Return JSON response
#         return jsonify({
#             'predicted_recliner_angle': predicted_angle,
#             'units': 'degrees',
#             'status': 'success'
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


# ++++++++++++++++++++++++++++++  tflite model ++++++++++++++++++++++++++++++

from flask import Flask, request, render_template, jsonify
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
import os
import requests

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_METAL_DISABLE"] = "1"
tf.config.set_visible_devices([], 'GPU')


def configure_tensorflow():
    """Configure TensorFlow to avoid Metal conflicts"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Option 1: Disable GPU entirely
    tf.config.set_visible_devices([], 'GPU')

    # Option 2: Configure GPU memory growth (if you need GPU)
    # try:
    #     gpus = tf.config.list_physical_devices('GPU')
    #     if gpus:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    # except RuntimeError:
    #     pass


# Call this before any other TensorFlow operations
configure_tensorflow()

app = Flask(__name__)

# Global variables
scaler = None
interpreter = None
input_details = None
output_details = None


def load_model_and_scaler():
    global scaler, interpreter, input_details, output_details

    tflite_model_path = "new_model.tflite"
    scaler_path = "improved_scaler.pkl"

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model and scaler loaded!")


def preprocess_input(input_data):
    df = pd.DataFrame(input_data, index=[0])

    if 'BMI' not in df.columns or pd.isna(df['BMI'][0]):
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    expected_columns = [
        'Height', 'Weight', 'BMI', 'WorkSittingTime',
        'Gender_Male', 'Gender_Female',
        'CommuteType_Car', 'CommuteType_Train', 'CommuteType_Walk'
    ]

    for col in expected_columns:
        if col not in df.columns:
            if col.startswith('Gender_'):
                df['Gender_Male'] = 1
                df['Gender_Female'] = 0
            elif col.startswith('CommuteType_'):
                df['CommuteType_Car'] = 1
                df['CommuteType_Train'] = 0
                df['CommuteType_Walk'] = 0
            else:
                df[col] = 0

    df['Height_Weight_Interaction'] = df['Height'] * df['Weight']
    df['BMI_SittingTime_Interaction'] = df['BMI'] * df['WorkSittingTime']

    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled = scaler.transform(df)

    return scaled.astype(np.float32)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/health')
def health_check():
    return {"status": "healthy"}, 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight']),
            'WorkSittingTime': float(request.form['sitting_time']),
            'Gender_Male': 1 if request.form['gender'] == 'male' else 0,
            'Gender_Female': 1 if request.form['gender'] == 'female' else 0,
            'CommuteType_Car': 1 if request.form['commute_type'] == 'car' else 0,
            'CommuteType_Train': 1 if request.form['commute_type'] == 'train' else 0,
            'CommuteType_Walk': 1 if request.form['commute_type'] == 'walk' else 0
        }

        processed = preprocess_input(input_data)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_angle = float(prediction[0][0])

        return render_template('index.html',
                               prediction_text=f'Recommended recliner angle: {predicted_angle:.1f}°',
                               show_result=True)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}',
                               show_result=True)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        required_fields = ['height', 'weight', 'work_sitting_time']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        input_data = {
            'Height': float(data['height']),
            'Weight': float(data['weight']),
            'WorkSittingTime': float(data['work_sitting_time']),
            'Gender_Male': 1 if data.get('gender', 'male').lower() == 'male' else 0,
            'Gender_Female': 1 if data.get('gender', 'male').lower() == 'female' else 0,
            'CommuteType_Car': 1 if data.get('commute_type', 'car').lower() == 'car' else 0,
            'CommuteType_Train': 1 if data.get('commute_type', 'car').lower() == 'train' else 0,
            'CommuteType_Walk': 1 if data.get('commute_type', 'car').lower() == 'walk' else 0
        }

        processed = preprocess_input(input_data)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_angle = float(prediction[0][0])

        return jsonify({
            'predicted_recliner_angle': predicted_angle,
            'units': 'degrees',
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    load_model_and_scaler()
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
