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

# ============================================================
# COMPONENT-BASED FAILURE INJECTION SYSTEM
# Maps physical truck components to their associated sensors
# ============================================================

# Component → Sensor Index Mapping
# Based on sensor_cols order: ['Speed_kmph', 'Load_tons', 'Fuel_Rate_Lph', 'Brake_Temp_C',
#                              'Vibration_mm_s', 'Oil_Pressure_bar', 'Hydraulic_Pressure_bar', 'Engine_Temp_C']
COMPONENT_SENSOR_MAP = {
    'engine': [7, 5, 2],      # Engine_Temp_C (idx 7), Oil_Pressure_bar (idx 5), Fuel_Rate_Lph (idx 2)
    'hydraulic': [6, 1],      # Hydraulic_Pressure_bar (idx 6), Load_tons (idx 1)
    'wheels': [4, 3, 0],      # Vibration_mm_s (idx 4), Brake_Temp_C (idx 3), Speed_kmph (idx 0)
    'chassis': [4, 1]         # Vibration_mm_s (idx 4), Load_tons (idx 1) - structural stress
}

# ============================================================
# WEIGHTED FAILURE PROBABILITIES
# Based on real-world industrial failure statistics:
# - Engine: High thermal/mechanical stress → highest failure rate
# - Hydraulics: Fluid systems prone to leaks → medium-high
# - Wheels: Wear items but robust → medium
# - Chassis: Structural, rarely fails → lowest
# ============================================================
COMPONENT_FAILURE_WEIGHTS = {
    'engine': 0.40,      # 40% - Highest: thermal stress, moving parts, combustion
    'hydraulic': 0.30,   # 30% - Medium-high: fluid leaks, seal wear, pressure issues
    'wheels': 0.20,      # 20% - Medium: brake wear, bearing issues, tire problems
    'chassis': 0.10      # 10% - Lowest: structural, rarely fails unless extreme stress
}

# Failure scenarios with probabilities derived from component weights
# Single failures are more common (70%), dual failures less common (30%)
FAILURE_SCENARIOS = [
    # Single component failures (70% total)
    (['engine'], 0.30),                    # Engine-only (highest weight component)
    (['hydraulic'], 0.22),                 # Hydraulic-only
    (['wheels'], 0.13),                    # Wheel/brake failure
    (['chassis'], 0.05),                   # Chassis-only (rare)
    # Dual component failures (30% total) - correlated failures
    (['engine', 'hydraulic'], 0.12),       # Engine overheating affects hydraulic fluid
    (['wheels', 'chassis'], 0.10),         # Wheel issues stress chassis
    (['hydraulic', 'wheels'], 0.08)        # Hydraulic brake system failure
]

# Global state for anomaly injection
injection_state = {
    'inject_anomaly': False,
    'active_failures': [],      # Currently failing components (stable during injection)
    'failure_locked': False,    # Lock failure selection while injection is ON
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

import random

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

def select_failure_scenario():
    """
    Select a realistic failure scenario based on weighted probabilities.
    Returns a list of component names that will fail.
    """
    rand = random.random()
    cumulative = 0.0
    for components, probability in FAILURE_SCENARIOS:
        cumulative += probability
        if rand <= cumulative:
            return components.copy()
    # Fallback to single engine failure
    return ['engine']

@app.route('/toggle_injection', methods=['POST'])
def toggle_injection():
    """Toggle anomaly injection state with component-based failure selection"""
    with injection_state['lock']:
        injection_state['inject_anomaly'] = not injection_state['inject_anomaly']
        current_state = injection_state['inject_anomaly']
        
        if current_state:
            # Injection turned ON: Select random failure scenario
            injection_state['active_failures'] = select_failure_scenario()
            injection_state['failure_locked'] = True
            print(f"[INJECTION ON] Failing components: {injection_state['active_failures']}")
        else:
            # Injection turned OFF: Clear failures
            injection_state['active_failures'] = []
            injection_state['failure_locked'] = False
            print("[INJECTION OFF] All components normal")
    
    return jsonify({
        'inject_anomaly': current_state,
        'failed_components': injection_state['active_failures']
    })

@app.route('/get_injection_state')
def get_injection_state():
    """Get current injection state including active failures"""
    with injection_state['lock']:
        return jsonify({
            'inject_anomaly': injection_state['inject_anomaly'],
            'failed_components': injection_state['active_failures'],
            'failure_probabilities': COMPONENT_FAILURE_WEIGHTS
        })

@app.route('/get_failure_probabilities')
def get_failure_probabilities():
    """
    Return the failure probability weights for all components.
    This endpoint provides static probability data for UI display.
    """
    return jsonify({
        'probabilities': COMPONENT_FAILURE_WEIGHTS,
        'description': {
            'engine': 'High thermal/mechanical stress - most failure-prone',
            'hydraulic': 'Fluid system leaks and seal wear',
            'wheels': 'Brake wear and bearing issues',
            'chassis': 'Structural - rarely fails'
        }
    })

@app.route('/simulate_data')
def simulate_data():
    """
    Generate and analyze simulated sensor data with COMPONENT-BASED failure injection.
    Only sensors belonging to failed components will spike during injection.
    """
    global data_buffer
    
    # Get injection state and active failures
    with injection_state['lock']:
        inject = injection_state['inject_anomaly']
        active_failures = injection_state['active_failures'].copy()
    
    # Generate base sensor data (normal distribution for all sensors)
    means = np.array(data_stats['mean'])
    stds = np.array(data_stats['std'])
    raw_data = means + np.random.randn(len(means)) * stds * 0.3
    
    # ============================================================
    # COMPONENT-BASED FAILURE INJECTION
    # Only spike sensors belonging to the selected failed components
    # ============================================================
    if inject and active_failures:
        # Collect all sensor indices that should be spiked
        sensors_to_spike = set()
        for component in active_failures:
            if component in COMPONENT_SENSOR_MAP:
                sensors_to_spike.update(COMPONENT_SENSOR_MAP[component])
        
        # Spike ONLY the affected sensors by +8 to +12 standard deviations
        for sensor_idx in sensors_to_spike:
            spike_magnitude = 8 + np.random.rand() * 4  # Random between 8-12 std
            raw_data[sensor_idx] = means[sensor_idx] + spike_magnitude * stds[sensor_idx]
            # Add slight noise for realism
            raw_data[sensor_idx] += np.random.randn() * stds[sensor_idx] * 0.3
    
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
    
    # ============================================================
    # COMPONENT STATUS DETERMINATION
    # Check each component based on its sensor values vs thresholds
    # ============================================================
    affected_components = {
        'engine': False,
        'hydraulic': False,
        'chassis': False,
        'wheels': False
    }
    
    if is_anomaly:
        # Threshold multiplier for detecting component-level anomalies
        ANOMALY_THRESHOLD_MULTIPLIER = 3.0
        
        # ENGINE: Check Engine_Temp_C, Oil_Pressure_bar, Fuel_Rate_Lph
        engine_temp = raw_data[7]
        oil_pressure = raw_data[5]
        fuel_rate = raw_data[2]
        if (engine_temp > means[7] + ANOMALY_THRESHOLD_MULTIPLIER * stds[7] or
            oil_pressure > means[5] + ANOMALY_THRESHOLD_MULTIPLIER * stds[5] or
            fuel_rate > means[2] + ANOMALY_THRESHOLD_MULTIPLIER * stds[2]):
            affected_components['engine'] = True
        
        # HYDRAULIC: Check Hydraulic_Pressure_bar, Load_tons
        hydraulic_pressure = raw_data[6]
        load = raw_data[1]
        if (hydraulic_pressure > means[6] + ANOMALY_THRESHOLD_MULTIPLIER * stds[6] or
            load > means[1] + ANOMALY_THRESHOLD_MULTIPLIER * stds[1]):
            affected_components['hydraulic'] = True
        
        # WHEELS: Check Vibration_mm_s, Brake_Temp_C, Speed_kmph
        vibration = raw_data[4]
        brake_temp = raw_data[3]
        speed = raw_data[0]
        if (vibration > means[4] + ANOMALY_THRESHOLD_MULTIPLIER * stds[4] or
            brake_temp > means[3] + ANOMALY_THRESHOLD_MULTIPLIER * stds[3] or
            speed > means[0] + ANOMALY_THRESHOLD_MULTIPLIER * stds[0]):
            affected_components['wheels'] = True
        
        # CHASSIS: Check Vibration_mm_s, Load_tons (structural stress indicators)
        if (vibration > means[4] + ANOMALY_THRESHOLD_MULTIPLIER * stds[4] or
            load > means[1] + ANOMALY_THRESHOLD_MULTIPLIER * stds[1]):
            affected_components['chassis'] = True
    
    # Build response with enhanced component information
    response = {
        'sensor_readings': sensor_readings,
        'reconstruction_error': round(reconstruction_error, 4),
        'threshold': round(float(threshold), 4),
        'is_anomaly': bool(is_anomaly),
        'pca_coords': pca_coords,
        'affected_components': affected_components,
        'failed_components': active_failures,  # Currently failing components
        'failure_probabilities': COMPONENT_FAILURE_WEIGHTS,  # Static probability weights
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
