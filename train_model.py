class Config:
    # Data paths
    TRAIN_DATA_PATH = "train_data"  # Path to the training data directory
    TEST_DATA_PATH = "test_data"  # Path to the testing data directory

    # Audio parameters
    SAMPLE_RATE = 22050  # Sampling rate of audio files in Hertz
    DURATION = 3  # Length of audio to be clipped (in seconds)

    # Feature extraction parameters
    N_MFCC = 20  # Number of MFCC (Mel-frequency cepstral coefficients) features to extract
    N_MELS = 128  # Number of Mel filterbanks for spectrogram
    FMAX = 8000  # Maximum frequency for Mel filters in Hertz

    # Training parameters
    BATCH_SIZE = 32  # Batch size used during training
    EPOCHS = 50  # Number of epochs for training
    LEARNING_RATE = 0.001  # Learning rate for the optimizer

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
        # Initialize the ModelManager with the configuration object
        self.config = config
        self.models = self._initialize_models()  # Prepare the models

    def _initialize_models(self):
        # Define and initialize individual models for classification
        models = {
            'dt': DecisionTreeClassifier(
                max_depth=10,  # Limit the depth of the tree
                min_samples_split=5,  # Minimum samples required to split a node
                min_samples_leaf=2,  # Minimum samples in a leaf node
                class_weight='balanced',  # Adjust weights for imbalanced classes
                random_state=42  # Seed for reproducibility
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,  # Number of trees in the forest
                max_depth=10,  # Limit the depth of each tree
                min_samples_split=5,  # Minimum samples required to split a node
                min_samples_leaf=2,  # Minimum samples in a leaf node
                class_weight='balanced',  # Adjust weights for imbalanced classes
                random_state=42  # Seed for reproducibility
            ),
            'xgb': XGBClassifier(
                n_estimators=200,  # Number of boosting rounds
                max_depth=6,  # Maximum depth of a tree
                learning_rate=0.02,  # Learning rate for gradient boosting
                subsample=0.8,  # Fraction of samples used per boosting round
                colsample_bytree=0.8,  # Fraction of features used per tree
                random_state=42  # Seed for reproducibility
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,  # Number of neighbors to consider
                weights='distance',  # Weight by distance
                metric='minkowski',  # Distance metric (Minkowski distance)
                p=2  # Parameter for Minkowski distance (p=2 for Euclidean)
            ),
            'svm': SVC(
                kernel='rbf',  # Radial Basis Function kernel
                C=1.0,  # Regularization parameter
                gamma='scale',  # Kernel coefficient
                class_weight='balanced',  # Adjust weights for imbalanced classes
                probability=True,  # Enable probability estimates
                random_state=42  # Seed for reproducibility
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
            voting='soft',  # Use soft voting (probabilities)
            weights=[1, 2, 2, 1, 2]  # Assign higher weights to RF and XGB
        )

        models['ensemble'] = ensemble  # Add the ensemble model
        return models

    def train_evaluate(self, X_train, y_train, X_test, y_test):
        # Train and evaluate all models
        results = {}

        # Iterate through models with a progress bar
        for name, model in tqdm(self.models.items(), desc='Training models'):
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)  # Train the model

            y_pred = model.predict(X_test)  # Predict on the test set
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
            report = classification_report(y_test, y_pred)  # Generate classification report

            results[name] = {
                'accuracy': accuracy,  # Store accuracy
                'report': report  # Store classification report
            }

            print(f"{name} Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)

        return results  # Return the evaluation results


import os
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm


class DataLoader:
    def __init__(self, config):
        # Initialize DataLoader with configuration and feature extractor
        self.config = config
        self.feature_extractor = FeatureExtractor(config)  # Create feature extractor instance
        self.scaler = StandardScaler()  # Use standard scaling

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Load audio features and labels from the specified data path
        features = []  # List to store features
        labels = []  # List to store labels

        # Count total files for progress bar
        total_files = sum([len([f for f in os.listdir(os.path.join(data_path, c))
                                if f.endswith('.wav')])
                           for c in self.config.CLASSES])

        pbar = tqdm(total=total_files, desc='Loading audio files')  # Progress bar

        for class_name in self.config.CLASSES:  # Iterate through each class
            class_path = os.path.join(data_path, class_name)  # Get class directory
            for file_name in os.listdir(class_path):
                if file_name.endswith('.wav'):  # Process only .wav files
                    file_path = os.path.join(class_path, file_name)  # Construct file path
                    feature = self.feature_extractor.extract_features(file_path)  # Extract features
                    features.append(feature)  # Append features
                    labels.append(self.config.CLASSES.index(class_name))  # Append class index
                    pbar.update(1)  # Update progress bar

        pbar.close()  # Close progress bar

        X = np.array(features)  # Convert features list to numpy array
        y = np.array(labels)  # Convert labels list to numpy array

        # Standardize features
        if data_path == self.config.TRAIN_DATA_PATH:
            X = self.scaler.fit_transform(X)  # Fit and transform for training data
        else:
            X = self.scaler.transform(X)  # Transform for testing data

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
            n_mfcc=self.config.N_MFCC  # Number of MFCC features
        )

        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Centroids
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]  # Rolloff
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]  # Zero crossing rate

        # Compute statistics for MFCC
        mfcc_mean = np.mean(mfcc, axis=1)  # Mean
        mfcc_std = np.std(mfcc, axis=1)  # Standard deviation
        mfcc_25 = np.percentile(mfcc, 25, axis=1)  # 25th percentile
        mfcc_75 = np.percentile(mfcc, 75, axis=1)  # 75th percentile

        # Compute statistics for scalar features
        scalar_features = np.array([
            np.mean(spectral_centroids),  # Mean centroid
            np.std(spectral_centroids),  # Std centroid
            np.mean(spectral_rolloff),  # Mean rolloff
            np.std(spectral_rolloff),  # Std rolloff
            np.mean(zero_crossing_rate),  # Mean zero crossing rate
            np.std(zero_crossing_rate)  # Std zero crossing rate
        ])

        # Combine all features into a single feature vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_25, mfcc_75, scalar_features
        ])

        return features  # Return the combined feature vector


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


def save_results(validation_results, test_results):
    # Save validation and test results to a file
    if not os.path.exists('results'):
        os.makedirs('results')  # Create results directory if it doesn't exist

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate timestamp
    filename = f'results/classification_results_{timestamp}.txt'  # Results filename

    with open(filename, 'w', encoding='utf-8') as f:
        # Write validation results
        f.write("=== Validation Results ===\n\n")
        for model_name, result in validation_results.items():
            f.write(f"\n{model_name} Results:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")  # Write accuracy
            f.write("Classification Report:\n")
            f.write(result['report'])  # Write classification report
            f.write("\n" + "=" * 50 + "\n")

        # Write test results
        f.write("\n\n=== Test Results ===\n")
        for model_name, result in test_results.items():
            f.write(f"\n{model_name} Test Performance:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")  # Write accuracy
            f.write("Classification Report:\n")
            f.write(result['report'])  # Write classification report
            f.write("\n" + "=" * 50 + "\n")

    print(f"\nResults saved to: {filename}")


def save_model(model, scaler):
    # Save the best model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')  # Create models directory if it doesn't exist

    joblib.dump(model, 'models/best_model.pkl')  # Save the model
    joblib.dump(scaler, 'models/scaler.pkl')  # Save the scaler
    print("\nModel saved to: models/best_model.pkl")


def main():
    # Initialize configuration
    config = Config()

    # Load data
    print("Loading data...")
    data_loader = DataLoader(config)

    print("Loading training data...")
    X_train, y_train = data_loader.load_data(config.TRAIN_DATA_PATH)  # Load training data

    print("Loading testing data...")
    X_test, y_test = data_loader.load_data(config.TEST_DATA_PATH)  # Load testing data

    print("Splitting validation set...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,  # 20% for validation
        random_state=42,  # Seed for reproducibility
        stratify=y_train  # Preserve class distribution
    )

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    model_manager = ModelManager(config)  # Initialize ModelManager
    validation_results = model_manager.train_evaluate(X_train, y_train, X_val, y_val)  # Evaluate on validation set

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_results = {}

    for model_name, model in model_manager.models.items():  # Iterate through models
        print(f"\nEvaluating {model_name} on test data...")
        test_pred = model.predict(X_test)  # Predict on test set
        test_accuracy = accuracy_score(y_test, test_pred)  # Calculate accuracy
        test_report = classification_report(y_test, test_pred)  # Generate classification report

        test_results[model_name] = {
            'accuracy': test_accuracy,  # Store accuracy
            'report': test_report  # Store classification report
        }

        print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{model_name} Test Classification Report:")
        print(test_report)

    # Save results
    save_results(validation_results, test_results)  # Save validation and test results

    # Save the best model (e.g., ensemble)
    best_model_name = "ensemble"  # Best-performing model (default: ensemble)
    save_model(model_manager.models[best_model_name], data_loader.scaler)  # Save model and scaler


if __name__ == "__main__":
    main()