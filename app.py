"""
Dump Truck Engine Anomaly Detection - Flask Application
Real-time anomaly detection with autoencoder and live visualization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import threading
import time

app = Flask(__name__)

# Global state for anomaly injection
injection_state = {
    'inject_anomaly': False,
    'lock': threading.Lock()
}

# Data buffer for time series
data_buffer = {
    'engine_temp': [],
    'hydraulic_pressure': [],
    'vibration': [],
    'speed': [],
    'timestamps': [],
    'max_points': 250
}

# Load model artifacts
print("Loading model artifacts...")
try:
    model = keras.models.load_model('autoencoder_model.h5', compile=False)
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    threshold = joblib.load('threshold.pkl')
    data_stats = joblib.load('data_stats.pkl')
    print("All artifacts loaded successfully!")
    print(f"Threshold: {threshold}")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    print("Please run train_model.py first!")
    model = None
    scaler = None
    pca = None
    threshold = 0.5
    data_stats = {
        'mean': [35, 55, 45, 120, 2.5, 4.5, 210, 85],
        'std': [5, 10, 8, 15, 0.5, 0.8, 20, 10],
        'sensor_cols': ['Speed_kmph', 'Load_tons', 'Fuel_Rate_Lph', 'Brake_Temp_C',
                        'Vibration_mm_s', 'Oil_Pressure_bar', 'Hydraulic_Pressure_bar', 'Engine_Temp_C']
    }

# Sensor column mapping for display
SENSOR_DISPLAY_NAMES = {
    'Speed_kmph': 'SPEED KM/H',
    'Load_tons': 'LOAD TONS',
    'Fuel_Rate_Lph': 'FUEL RATE L/H',
    'Brake_Temp_C': 'BRAKE TEMP °C',
    'Vibration_mm_s': 'VIBRATION MM/S',
    'Oil_Pressure_bar': 'OIL PRESSURE BAR',
    'Hydraulic_Pressure_bar': 'HYDRAULIC PRESSURE BAR',
    'Engine_Temp_C': 'ENGINE TEMP °C'
}

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/toggle_injection', methods=['POST'])
def toggle_injection():
    """Toggle anomaly injection state"""
    with injection_state['lock']:
        injection_state['inject_anomaly'] = not injection_state['inject_anomaly']
        current_state = injection_state['inject_anomaly']
    return jsonify({'inject_anomaly': current_state})

@app.route('/get_injection_state')
def get_injection_state():
    """Get current injection state"""
    with injection_state['lock']:
        return jsonify({'inject_anomaly': injection_state['inject_anomaly']})

@app.route('/simulate_data')
def simulate_data():
    """Generate and analyze simulated sensor data"""
    global data_buffer
    
    # Get injection state
    with injection_state['lock']:
        inject = injection_state['inject_anomaly']
    
    # Generate sensor data
    means = np.array(data_stats['mean'])
    stds = np.array(data_stats['std'])
    
    if inject:
        # CRITICAL: Spike ALL sensor values by +10 standard deviations
        raw_data = means + 10 * stds + np.random.randn(len(means)) * stds * 0.5
    else:
        # Normal distribution
        raw_data = means + np.random.randn(len(means)) * stds * 0.3
    
    # Ensure positive values
    raw_data = np.maximum(raw_data, 0.1)
    
    # Create sensor readings dictionary
    sensor_cols = data_stats['sensor_cols']
    sensor_readings = {}
    for i, col in enumerate(sensor_cols):
        display_name = SENSOR_DISPLAY_NAMES.get(col, col)
        sensor_readings[display_name] = round(float(raw_data[i]), 2)
    
    # Scale data for model
    if model is not None and scaler is not None:
        try:
            X_scaled = scaler.transform(raw_data.reshape(1, -1))
            
            # Get reconstruction
            X_pred = model.predict(X_scaled, verbose=0)
            
            # Calculate reconstruction error
            reconstruction_error = float(np.mean(np.square(X_scaled - X_pred)))
            
            # Determine if anomaly
            is_anomaly = reconstruction_error > threshold
            
            # Get PCA coordinates for visualization
            pca_coords = pca.transform(X_scaled)[0].tolist()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            reconstruction_error = 0.0
            is_anomaly = False
            pca_coords = [0, 0, 0]
    else:
        reconstruction_error = 0.0
        is_anomaly = inject  # Fallback
        pca_coords = [np.random.randn() * 5, np.random.randn() * 5, np.random.randn() * 5]
    
    # Update time series buffer
    current_time = len(data_buffer['timestamps'])
    data_buffer['timestamps'].append(current_time)
    data_buffer['engine_temp'].append(raw_data[7])  # Engine_Temp_C
    data_buffer['hydraulic_pressure'].append(raw_data[6])  # Hydraulic_Pressure_bar
    data_buffer['vibration'].append(raw_data[4])  # Vibration_mm_s
    data_buffer['speed'].append(raw_data[0])  # Speed_kmph
    
    # Trim buffer
    max_pts = data_buffer['max_points']
    if len(data_buffer['timestamps']) > max_pts:
        data_buffer['timestamps'] = data_buffer['timestamps'][-max_pts:]
        data_buffer['engine_temp'] = data_buffer['engine_temp'][-max_pts:]
        data_buffer['hydraulic_pressure'] = data_buffer['hydraulic_pressure'][-max_pts:]
        data_buffer['vibration'] = data_buffer['vibration'][-max_pts:]
        data_buffer['speed'] = data_buffer['speed'][-max_pts:]
    
    # Determine which components are affected
    affected_components = {
        'engine': False,
        'hydraulic': False,
        'chassis': False,
        'wheels': False
    }
    
    if is_anomaly:
        # Check which sensors are anomalous
        engine_temp = raw_data[7]
        hydraulic_pressure = raw_data[6]
        vibration = raw_data[4]
        brake_temp = raw_data[3]
        
        if engine_temp > means[7] + 3 * stds[7]:
            affected_components['engine'] = True
        if hydraulic_pressure > means[6] + 3 * stds[6]:
            affected_components['hydraulic'] = True
        if vibration > means[4] + 3 * stds[4]:
            affected_components['wheels'] = True
            affected_components['chassis'] = True
        if brake_temp > means[3] + 3 * stds[3]:
            affected_components['wheels'] = True
    
    # Build response
    response = {
        'sensor_readings': sensor_readings,
        'reconstruction_error': round(reconstruction_error, 4),
        'threshold': round(float(threshold), 4),
        'is_anomaly': bool(is_anomaly),
        'pca_coords': pca_coords,
        'affected_components': affected_components,
        'inject_active': inject,
        'time_series': {
            'timestamps': data_buffer['timestamps'][-100:],
            'engine_temp': data_buffer['engine_temp'][-100:],
            'hydraulic_pressure': data_buffer['hydraulic_pressure'][-100:],
            'vibration': data_buffer['vibration'][-100:],
            'speed': data_buffer['speed'][-100:]
        }
    }
    
    return jsonify(response)

@app.route('/reset_buffer', methods=['POST'])
def reset_buffer():
    """Reset the data buffer"""
    global data_buffer
    data_buffer = {
        'engine_temp': [],
        'hydraulic_pressure': [],
        'vibration': [],
        'speed': [],
        'timestamps': [],
        'max_points': 250
    }
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
