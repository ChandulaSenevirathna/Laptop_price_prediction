from flask import Flask, request, render_template
import numpy as np
import joblib  # or your preferred method to load the model

app = Flask(__name__)

# Load your pre-trained model (adjust path and method as needed)
model = joblib.load('./linear_regression_model.pkl')  # Replace with the correct path to your model

@app.route('/')
def index():
    return render_template('laptop.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract form data
    data = {
        'Inches': float(request.form['inches']),
        'Ram': float(request.form['ram']),
        'Weight': float(request.form['weight']),
        'CPU Frequency': float(request.form['cpu_freq']),
        'Memory Amount': float(request.form['memory_amount']),
        'Acer': int(request.form.get('acer', 0)),
        'Apple': int(request.form.get('apple', 0)),
        'Asus': int(request.form.get('asus', 0)),
        'Chuwi': int(request.form.get('chuwi', 0)),
        'Dell': int(request.form.get('dell', 0)),
        'Fujitsu': int(request.form.get('fujitsu', 0)),
        'Google': int(request.form.get('google', 0)),
        'HP': int(request.form.get('hp', 0)),
        'Huawei': int(request.form.get('huawei', 0)),
        'LG': int(request.form.get('lg', 0)),
        'Lenovo': int(request.form.get('lenovo', 0)),
        'MSI': int(request.form.get('msi', 0)),
        'Mediacom': int(request.form.get('mediacom', 0)),
        'Microsoft': int(request.form.get('microsoft', 0)),
        'Razer': int(request.form.get('razer', 0)),
        'Samsung': int(request.form.get('samsung', 0)),
        'Toshiba': int(request.form.get('toshiba', 0)),
        'Vero': int(request.form.get('vero', 0)),
        'Xiaomi': int(request.form.get('xiaomi', 0)),
        '2 in 1 Convertible': int(request.form.get('convertible', 0)),
        'Gaming': int(request.form.get('gaming', 0)),
        'Netbook': int(request.form.get('netbook', 0)),
        'Notebook': int(request.form.get('notebook', 0)),
        'Ultrabook': int(request.form.get('ultrabook', 0)),
        'Workstation': int(request.form.get('workstation', 0)),
        'AMD_CPU': int(request.form.get('amd_cpu', 0)),
        'Intel_CPU': int(request.form.get('intel_cpu', 0)),
        'Samsung_CPU': int(request.form.get('samsung_cpu', 0)),
        'AMD_GPU': int(request.form.get('amd_gpu', 0)),
        'ARM_GPU': int(request.form.get('arm_gpu', 0)),
        'Intel_GPU': int(request.form.get('intel_gpu', 0)),
        'Nvidia_GPU': int(request.form.get('nvidia_gpu', 0)),
        'Android': int(request.form.get('android', 0)),
        'Chrome OS': int(request.form.get('chrome_os', 0)),
        'Linux': int(request.form.get('linux', 0)),
        'Mac OS X': int(request.form.get('mac_os_x', 0)),
        'No OS': int(request.form.get('no_os', 0)),
        'Windows 10': int(request.form.get('windows_10', 0)),
        'Windows 10 S': int(request.form.get('windows_10_s', 0)),
        'Windows 7': int(request.form.get('windows_7', 0)),
        'macOS': int(request.form.get('macos', 0)),
        'Flash': int(request.form.get('flash', 0)),
        'HDD': int(request.form.get('hdd', 0)),
        'Hybrid': int(request.form.get('hybrid', 0)),
        'SSD': int(request.form.get('ssd', 0))
    }
    

    # Convert the data to the format expected by your model
    features = np.array([
        data['Inches'],
        data['Ram'],
        data['Weight'],
        data['Acer'],
        data['Apple'],
        data['Asus'],
        data['Chuwi'],
        data['Dell'],
        data['Fujitsu'],
        data['Google'],
        data['HP'],
        data['Huawei'],
        data['LG'],
        data['Lenovo'],
        data['MSI'],
        data['Mediacom'],
        data['Microsoft'],
        data['Razer'],
        data['Samsung'],
        data['Toshiba'],
        data['Vero'],
        data['Xiaomi'],
        data['2 in 1 Convertible'],
        data['Gaming'],
        data['Netbook'],
        data['Notebook'],
        data['Ultrabook'],
        data['Workstation'],
        data['CPU Frequency'],
        data['AMD_CPU'],
        data['Intel_CPU'],
        data['Samsung_CPU'],
        data['Memory Amount'],
        data['AMD_GPU'],
        data['ARM_GPU'],
        data['Intel_GPU'],
        data['Nvidia_GPU'],
        data['Android'],
        data['Chrome OS'],
        data['Linux'],
        data['Mac OS X'],
        data['No OS'],
        data['Windows 10'],
        data['Windows 10 S'],
        data['Windows 7'],
        data['macOS'],
        data['Flash'],
        data['HDD'],
        data['Hybrid'],
        data['SSD']
    ]).reshape(1, -1)

    # Make prediction
    predicted_price = model.predict(features)[0]

    # Render the result
    return render_template('laptop.html', predicted_price=f"{predicted_price:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
