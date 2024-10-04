import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st


@st.cache_resource
def load_data_and_models():
# Đọc dữ liệu từ tệp
    data = pd.read_csv('./dataa.csv')

# Tạo cột 'Tien_dien' (giả sử đây là tổng công suất hoạt động nhân với cường độ)
    data['Tien_dien'] = data['Cong_suat_hoat_dong_toan_cau'] * data['Cuong_do_toan_cau']

# Tách dữ liệu thành features và target
    X = data[['Cong_suat_hoat_dong_toan_cau', 'Cuong_do_toan_cau']]
    y = data['Tien_dien']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


   # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
# Hàm tính Nash-Sutcliffe Efficiency (NSE)
    def nash_sutcliffe_efficiency(obs, pred):
        return 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

# Đánh giá mô hình
    def evaluate_model(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        nse = nash_sutcliffe_efficiency(y_test, y_pred)
        return r2, rmse, mae, nse

# 1. Mô hình Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    r2_linear, rmse_linear, mae_linear, nse_linear = evaluate_model(y_test, y_pred_linear)


# 2. Mô hình Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    r2_ridge, rmse_ridge, mae_ridge, nse_ridge = evaluate_model(y_test, y_pred_ridge)

# Khởi tạo mô hình
    neural_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)

# Huấn luyện mô hình
    neural_model.fit(X_train, y_train)

# Dự đoán
    y_pred_neural = neural_model.predict(X_test)
    r2_neural, rmse_neural, mae_neural, nse_neural = evaluate_model(y_test, y_pred_neural)

# 3. Mô hình Stacking Regression
    estimators = [
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('neural',MLPRegressor()),

]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=KNeighborsRegressor())
    stacking_model.fit(X_train, y_train)
    y_pred_stacking = stacking_model.predict(X_test)
    r2_stacking, rmse_stacking, mae_stacking, nse_stacking = evaluate_model(y_test, y_pred_stacking)

# In kết quả
    print("Linear Regression")
    print(f"R²: {r2_linear:.4f}, RMSE: {rmse_linear:.4f}, MAE: {mae_linear:.4f}, NSE: {nse_linear:.4f}")
    print("Ridge Regression")
    print(f"R²: {r2_ridge:.4f}, RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}, NSE: {nse_ridge:.4f}")
    print("Neural network")
    print(f"R²: {r2_neural:.4f}, RMSE: {rmse_neural:.4f}, MAE: {mae_neural:.4f}, NSE: {nse_neural:.4f}")
    print("Stacking Regression")
    print(f"R²: {r2_stacking:.4f}, RMSE: {rmse_stacking:.4f}, MAE: {mae_stacking:.4f}, NSE: {nse_stacking:.4f}")

    def plot_predictions(y_test, y_pred, model_name):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Dự đoán')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Đường lý tưởng')
        plt.xlabel('Giá trị thực tế')
        plt.ylabel('Giá trị dự đoán')
        plt.title(f'Biểu đồ dự đoán của mô hình {model_name}')
        plt.legend()
        plt.show()

# 1. Biểu đồ cho Linear Regression
    plot_predictions(y_test, y_pred_linear, 'Linear Regression')

# 2. Biểu đồ cho Ridge Regression
    plot_predictions(y_test, y_pred_ridge, 'Ridge Regression')

#Biểu đồ cho Neural network
    plot_predictions(y_test, y_pred_neural, 'Neural Netwok')

# 3. Biểu đồ cho Stacking Regression
    plot_predictions(y_test, y_pred_stacking, 'Stacking Regression')

# Chỉ tải và khởi tạo mô hình một lần
scaler, linear_model,ridge_model, neural_model, stacking_model, X_columns = load_data_and_models()

# Định nghĩa hàm dự đoán
def predict_price(model, dataset):
    input_data = pd.DataFrame([dataset])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    
    # Kiểm tra dữ liệu đầu vào
    if input_data_scaled.shape[1] != len(X_columns):
        st.error("Dữ liệu đầu vào không hợp lệ!")
        return None
    
    return model.predict(input_data_scaled)[0]

# Giao diện người dùng
st.title("Dự đoán mức tiêu thụ điện ")
st.write("Chọn mô hình .")

# Nút chọn mô hình
model_options = {
    'Linear Regression': linear_model,
    'Ridge Regression': ridge_model,
    'Neural Network': neural_model,
    'Stacking Model': stacking_model
}

selected_model_name = st.selectbox("Chọn mô hình:", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Nhập dữ liệu từ người dùng
cshoatdong = st.number_input('Cong_suat_hoat_dong_toan_cau', value=2.58)
cuongdo = st.number_input('Cuong_do_toan_cau', value=10.6)


new_data = {
    'Cong_suat_hoat_dong_toan_cau':cshoatdong,
    'Cuong_do_toan_cau':cuongdo,
   
}

if st.button("Dự đoán"):
    predicted_price = predict_price(selected_model, new_data)
    if predicted_price is not None:
        st.write(f"dự đoán: ${predicted_price:,.2f}")
