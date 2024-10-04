from flask import Flask, jsonify, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(model_dir, 'ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking_regressor_model.joblib'))


@app.route('/api/predict', methods=['POST'])
def prediction():
    data = request.json  # Lấy dữ liệu từ frontend
    active_power = float(data['active_power'])
    reactive_power = float(data[' reactive_power'])
    voltage = float(data['voltage'])
    intensity = float(data['intensity'])
    model_type = data['model']

    # Chuyển đổi dữ liệu thành định dạng mà mô hình yêu cầu
    features = np.array([[active_power, reactive_power, voltage, intensity]])

 # Lấy tên mô hình được chọn
    model_name = data['model']

        
        # Lấy mô hình tương ứng
    model = model.get(model_name)

    if model_type == "linear_regression":
        model = lr_model
    elif model_type == "ridge_regression":
        model = ridge_model
    elif model_type == "Stacking":
        model = stacking_model
    else:
        return jsonify({"error": "Invalid model type"}), 400

    # Thực hiện dự đoán
    prediction = model.predict(features)

    # Trả về kết quả
    return jsonify({"prediction": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)