import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("=== STREEVA: Training Activity State Classifier ===")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "UCI_HAR_Dataset.zip")
    extract_path = data_dir
    har_dir = os.path.join(extract_path, "UCI HAR Dataset")
    
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    tflite_path = os.path.join(models_dir, "motion_anomaly.tflite")
    
    # 1. Download & Extract Dataset
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    if not os.path.exists(zip_path):
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete.")
        
    if not os.path.exists(har_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")
        
    # 2. Label Mapping
    # Original labels: 1=Walking, 2=Upstairs, 3=Downstairs, 4=Sitting, 5=Standing, 6=Laying
    def map_labels(y_raw):
        y_raw = y_raw.ravel()
        # 4, 5, 6 -> 0 (Steady)
        # 1, 2, 3 -> 1 (Dynamic)
        return np.where(np.isin(y_raw, [4, 5, 6]), 0, 1)

    print("Loading labels...")
    y_train_raw = pd.read_csv(os.path.join(har_dir, 'train', 'y_train.txt'), header=None, sep=r'\s+').values
    y_test_raw = pd.read_csv(os.path.join(har_dir, 'test', 'y_test.txt'), header=None, sep=r'\s+').values
    
    y_train_binary = map_labels(y_train_raw)
    y_test_binary = map_labels(y_test_raw)
    
    print(f"Training samples: {len(y_train_binary)} (Steady: {sum(y_train_binary==0)}, Dynamic: {sum(y_train_binary==1)})")
    print(f"Test samples: {len(y_test_binary)} (Steady: {sum(y_test_binary==0)}, Dynamic: {sum(y_test_binary==1)})")
    
    # 3. Load Raw Inertial Signals
    def load_signals(subset):
        signals = []
        for axis in ['total_acc_x', 'total_acc_y', 'total_acc_z']:
            filename = f'{axis}_{subset}.txt'
            path = os.path.join(har_dir, subset, 'Inertial Signals', filename)
            df = pd.read_csv(path, header=None, sep=r'\s+')
            signals.append(df.values)
        # Stack into shape: (samples, timesteps, features)
        return np.dstack(signals)
        
    print("Loading raw inertial signals...")
    X_train_raw = load_signals('train')
    X_test_raw = load_signals('test')
    print(f"Raw Train Shape: {X_train_raw.shape}")
    
    # 4. Define and Train 1D-CNN
    print("Building 1D-CNN Model...")
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 3)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(
        X_train_raw, y_train_binary,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 5. Evaluate
    print("Evaluating model...")
    cnn_probs = model.predict(X_test_raw)
    cnn_preds = (cnn_probs > 0.5).astype(int).flatten()
    
    print("\n1D-CNN Results (Steady vs Dynamic):")
    print(classification_report(y_test_binary, cnn_preds, target_names=['Steady (0)', 'Dynamic (1)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_binary, cnn_preds))
    
    # 6. Export to TFLite
    print(f"Exporting to TFLite: {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Model exported successfully. File size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
