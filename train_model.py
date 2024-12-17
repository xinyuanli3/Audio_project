class Config:
    # Data paths
    TRAIN_DATA_PATH = "train_data"
    TEST_DATA_PATH = "test_data"

    # Audio parameters
    SAMPLE_RATE = 22050  # rate of audio
    DURATION = 3  # Length of audio (in seconds)

    # Feature extraction parameters
    N_MFCC = 20
    N_MELS = 128
    FMAX = 8000

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Audio classification
    CLASSES = ['table', 'water', 'sofa', 'glass', 'blackboard', 'railing', 'ben']


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tqdm import tqdm


class ModelManager:
    def __init__(self, config):
        # Initialize with the config
        self.config = config
        self.models = self._initialize_models()

    def _initialize_models(self):
        # Models for classification
        models = {
            'dt': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'xgb': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski',
                p=2
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }

        # Create an ensemble model using weighted voting
        ensemble = VotingClassifier(
            estimators=[
                ('dt', models['dt']),
                ('rf', models['rf']),
                ('xgb', models['xgb']),
                ('knn', models['knn']),
                ('svm', models['svm'])
            ],
            voting='soft',
            weights=[1, 2, 2, 1, 2]  # Assign higher weights to RF , XGB and SVM
        )

        # Add the ensemble model
        models['ensemble'] = ensemble
        return models

    def train_evaluate(self, X_train, y_train, X_test, y_test):
        # Train and evaluate all models
        results = {}

        # Iterate through models with a progress bar
        for name, model in tqdm(self.models.items(), desc='Training models'):
            print(f"\nTraining {name}...")
            # Train the model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # Generate classification report
            report = classification_report(y_test, y_pred)

            results[name] = {
                'accuracy': accuracy,
                'report': report
            }

            print(f"{name} Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)

        return results


import os
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm


class DataLoader:
    def __init__(self, config):
        # Initialize DataLoader with config and feature extractor
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.scaler = StandardScaler()

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Load audio features and labels
        features = []
        labels = []

        # Count total files for progress bar
        total_files = sum([len([f for f in os.listdir(os.path.join(data_path, c))
                                if f.endswith('.wav')])
                           for c in self.config.CLASSES])

        pbar = tqdm(total=total_files, desc='Loading audio files')

        for class_name in self.config.CLASSES:
            # Get class directory
            class_path = os.path.join(data_path, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.wav'):  # Process only .wav files
                    file_path = os.path.join(class_path, file_name)  # Construct file path
                    feature = self.feature_extractor.extract_features(file_path)  # Extract features
                    features.append(feature)  # Append features
                    labels.append(self.config.CLASSES.index(class_name))  # Append class index
                    pbar.update(1)  # Update progress bar

        pbar.close()

        X = np.array(features)
        y = np.array(labels)

        # Standardize features
        if data_path == self.config.TRAIN_DATA_PATH:
            # Fit and transform for training data
            X = self.scaler.fit_transform(X)
        else:
            # Transform for testing data
            X = self.scaler.transform(X)

        return X, y  # Return features and labels


import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        # Initialize the feature extractor with configuration
        self.config = config

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from an audio file"""
        # Load audio file
        y, sr = librosa.load(
            audio_path,
            duration=self.config.DURATION,  # Clip audio to specified duration
            sr=self.config.SAMPLE_RATE  # Resample to specified rate
        )

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.N_MFCC
        )

        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

        # Compute statistics for MFCC
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_25 = np.percentile(mfcc, 25, axis=1)
        mfcc_75 = np.percentile(mfcc, 75, axis=1)

        # Compute statistics for scalar features
        scalar_features = np.array([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])

        # Combine all features into a single feature vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_25, mfcc_75, scalar_features
        ])

        return features


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


def save_results(validation_results, test_results):
    # Save validation and test results to a file
    if not os.path.exists('results'):
        os.makedirs('results')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/classification_results_{timestamp}.txt'

    with open(filename, 'w', encoding='utf-8') as f:
        # Write validation results
        f.write("=== Validation Results ===\n\n")
        for model_name, result in validation_results.items():
            f.write(f"\n{model_name} Results:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(result['report'])
            f.write("\n" + "=" * 50 + "\n")

        # Write test results
        f.write("\n\n=== Test Results ===\n")
        for model_name, result in test_results.items():
            f.write(f"\n{model_name} Test Performance:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(result['report'])
            f.write("\n" + "=" * 50 + "\n")

    print(f"\nResults saved to: {filename}")


def save_model(model, scaler):
    # Save the best model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    # Save the model
    joblib.dump(model, 'models/best_model.pkl')
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModel saved to: models/best_model.pkl")


def main():
    # Initialize configuration
    config = Config()

    # Load data
    print("Loading data...")
    data_loader = DataLoader(config)

    print("Loading training data...")
    X_train, y_train = data_loader.load_data(config.TRAIN_DATA_PATH)

    print("Loading testing data...")
    X_test, y_test = data_loader.load_data(config.TEST_DATA_PATH)

    print("Splitting validation set...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,  # 20% for validation
        random_state=42,
        stratify=y_train
    )

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    model_manager = ModelManager(config)
    validation_results = model_manager.train_evaluate(X_train, y_train, X_val, y_val)

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_results = {}

    for model_name, model in model_manager.models.items():
        print(f"\nEvaluating {model_name} on test data...")
        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_report = classification_report(y_test, test_pred)

        test_results[model_name] = {
            'accuracy': test_accuracy,
            'report': test_report
        }

        print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{model_name} Test Classification Report:")
        print(test_report)

    # Save results
    save_results(validation_results, test_results)

    # Save the best model (e.g., ensemble)
    best_model_name = "ensemble"
    save_model(model_manager.models[best_model_name], data_loader.scaler)


if __name__ == "__main__":
    main()