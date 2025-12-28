"""
Dump Truck Engine Anomaly Detection - Model Training
Trains an Autoencoder on NORMAL sensor data to detect anomalies
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import joblib
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_data(filepath):
    """Load dataset and prepare features"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Define sensor columns (matching actual dataset column names)
    sensor_cols = [
        'speed_kmph', 'load_tons', 'fuel_rate_lph', 'brake_temp_C',
        'vibration_mm_s', 'oil_pressure_bar', 'hydraulic_pressure_bar', 'engine_temp_C'
    ]
    
    # Extract features
    X = df[sensor_cols].values
    
    # Filter normal data only (assuming 'label' column exists, 0 = normal)
    if 'label' in df.columns:
        normal_mask = df['label'] == 0
        X_normal = X[normal_mask]
        print(f"Total samples: {len(X)}, Normal samples: {len(X_normal)}")
    else:
        X_normal = X
        print(f"Total samples: {len(X)}")
    
    return X_normal, sensor_cols

def create_autoencoder(input_dim, encoding_dim=4):
    """Create autoencoder model"""
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(x)
    
    # Decoder
    x = layers.Dense(16, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Full autoencoder
    autoencoder = Model(inputs, decoded, name='autoencoder')
    
    # Encoder only (for feature extraction)
    encoder = Model(inputs, encoded, name='encoder')
    
    return autoencoder, encoder

def train_model():
    """Main training function"""
    print("=" * 60)
    print("DUMP TRUCK ENGINE ANOMALY DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X_normal, sensor_cols = load_and_prepare_data('dataset.csv.csv')
    
    # Scale the data
    print("\nScaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    
    # Calculate and save data statistics for simulation
    data_stats = {
        'mean': X_normal.mean(axis=0).tolist(),
        'std': X_normal.std(axis=0).tolist(),
        'sensor_cols': sensor_cols
    }
    joblib.dump(data_stats, 'data_stats.pkl')
    print("Data statistics saved to data_stats.pkl")
    
    # Create and compile autoencoder
    print("\nCreating autoencoder model...")
    input_dim = X_scaled.shape[1]
    autoencoder, encoder = create_autoencoder(input_dim, encoding_dim=4)
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    autoencoder.summary()
    
    # Train the model
    print("\nTraining autoencoder on normal data...")
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
    )
    
    # Calculate reconstruction errors on training data
    print("\nCalculating reconstruction errors...")
    X_pred = autoencoder.predict(X_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
    
    # Set threshold at 99th percentile of normal reconstruction errors
    threshold = np.percentile(reconstruction_errors, 99)
    print(f"Reconstruction error statistics:")
    print(f"  Mean: {np.mean(reconstruction_errors):.6f}")
    print(f"  Std:  {np.std(reconstruction_errors):.6f}")
    print(f"  Max:  {np.max(reconstruction_errors):.6f}")
    print(f"  Threshold (99th percentile): {threshold:.6f}")
    
    # Fit PCA for 3D visualization
    print("\nFitting PCA for visualization...")
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Save all artifacts
    print("\nSaving model artifacts...")
    
    # Save autoencoder model
    autoencoder.save('autoencoder_model.h5')
    print("Model saved to autoencoder_model.h5")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")
    
    # Save PCA
    joblib.dump(pca, 'pca.pkl')
    print("PCA saved to pca.pkl")
    
    # Save threshold
    joblib.dump(threshold, 'threshold.pkl')
    print("Threshold saved to threshold.pkl")
    
    # Verification
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - VERIFICATION")
    print("=" * 60)
    
    # Test on a sample
    sample = X_scaled[0:1]
    pred = autoencoder.predict(sample, verbose=0)
    error = np.mean(np.square(sample - pred))
    pca_coords = pca.transform(sample)
    
    print(f"Sample reconstruction error: {error:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Is anomaly: {error > threshold}")
    print(f"PCA coordinates: {pca_coords[0]}")
    
    # Test anomaly detection
    print("\nTesting anomaly injection...")
    anomaly_sample = X_normal[0:1].copy()
    anomaly_sample = anomaly_sample + 10 * X_normal.std(axis=0)  # Add 10 std devs
    anomaly_scaled = scaler.transform(anomaly_sample)
    anomaly_pred = autoencoder.predict(anomaly_scaled, verbose=0)
    anomaly_error = np.mean(np.square(anomaly_scaled - anomaly_pred))
    
    print(f"Anomaly sample reconstruction error: {anomaly_error:.6f}")
    print(f"Is detected as anomaly: {anomaly_error > threshold}")
    
    print("\n" + "=" * 60)
    print("All artifacts saved successfully!")
    print("Ready for deployment.")
    print("=" * 60)

if __name__ == '__main__':
    train_model()
