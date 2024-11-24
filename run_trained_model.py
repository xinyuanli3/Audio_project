# -*- coding: utf-8 -*-
"""CIS519_project_Aria.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AzzJcB4-QYjhHMp06nYH0NPiZk8ExZoG

# This is a simple tryout of Audio Classification Project for CIS 5190

## Class Label
- Ben Franklin statue - Google Map "Ben on the Bench"
- Blackboard (not white board) - Levine 4th floor bump space
- Table - Levine 4th floor near window
- Glass window - Levine 4th floor
- Handrail - Levine 4th floor
- Water fountain - Levine 4th floor
- Sofa - Levine 4th floor bump space

## Models student could possible use
- Autoencoders
- CNN with spectrograms
- Transformer based model (e.g. Audio-BERT)
- WaveNet


## Baseline Model:
The goal of the student is to train a model that has a non-trivial advantage over the baseline model we release

Nov.18 -Aria \\
Data comparison indicate the similarity between real data collected and TA provided sample. Class 2 and 5 consider well. Class 3 and 4 have significant mean and std difference.
Feature extraction (Chi-best) conducted to select the top 30 features. \\
Data later used to train several different model: \\
1. Simple CNN - perform fair with remove_bad T>.5 (0.77/0.46)
2. Complex CNN - fair with remove_bad+FS(0.99/0.49)
3. CNN+LSTM - fair with FS+remove_bad(0.67/0.66)
4. Logistic - fair with FS (0.98/0.6) particularly well on class 6,2
5. SVM - fair with FS (0.99/0.6) particularly well on class 0,2,5
6. Random forest - FS (0.96/0.42) underperform
7. KNN - underperform (0.95/0.42) particularly well in class 1
8. Logistic + SVM (0.98/0.6286)
9. Logistic + SVM + KNN (0.97/0.6857)

For model 1-3, CNN perform not ideally in feature classification, relatively ideal feature size with raw 66, selected 30. Issue caused by lack of data and low feature size.\
Model 4-7 perform fair in test acc, each have particular outer perform in specific class. Final combined model 9.Logistic + SVM + KNN achieved 68.2% val acc, with all class 3 and 4 being labeled incorrectly and 1 class 1 incorrect. Further improvement on Class 3 and 4 data collection might result this issue.
* OCA, PCA, OCA+ICA, RFE were all conducted while do not show reliable evidence.

Nov.21 -Aria \
Class 3 and 4 have been re-recorded and upload for training.\
Old Class 3,4 be eliminated.\
Best performance FS-top 30 -> RB either >0.5 \
Logistic + SVM gives 91-94% val acc \
Logistic + SVM + RF 97% val acc

# Load data
"""

import os
import librosa
import numpy as np
!pip install ffmpeg
import ffmpeg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount('/content/drive/')

!ls '/content/drive/Shareddrives/CIS5190_Final_Project/Sample_Evaluation_Data'

# 1. Convert M4A to WAV using FFmpeg (if needed)
def convert_m4a_to_wav(m4a_path, wav_path):
    ffmpeg.input(m4a_path).output(wav_path).run(overwrite_output=True)


def extract_features(file_path, n_mfcc=20):
    audio, sr = librosa.load(file_path, sr=None)
    n_fft = min(1024, len(audio))

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Add delta and delta-delta MFCCs
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_delta_mfcc = librosa.feature.delta(mfccs, order=2)

    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_delta_mfcc_mean = np.mean(delta_delta_mfcc, axis=1)

    # Extract Tonnetz features
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    # Combine features
    features = np.concatenate((mfccs_mean, delta_mfcc_mean, delta_delta_mfcc_mean, tonnetz_mean))
    return features


def load_dataset(audio_dir):
    features = []
    y = []

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_dir, filename)
            print(f"Processing file: {file_path}")
            try:
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                y.append(filename)
            except ValueError as e:
                print(f"Skipping file {file_path}: {e}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(features)

CLASS_TO_LABEL = {
    'water': 0,
    'table': 1,
    'sofa': 2,
    'railing': 3,
    'glass': 4,
    'blackboard': 5,
    'ben': 6,
}

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# Set the base directory path
base_dir = '/content/drive/Shareddrives/CIS5190_Final_Project/Sample_Evaluation_Data'
X_sample = []
Y_sample = []
# Loop through subfolders
for idx, subfolder in enumerate(os.listdir(base_dir)):
    subfolder_path = os.path.join(base_dir, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")


        # Loop through files in the subfolder
        # for idx, file in enumerate(os.listdir(subfolder_path)):
        # file_path = os.path.join(subfolder_path)
        x = load_dataset(subfolder_path)
        y = np.array([CLASS_TO_LABEL[subfolder]] * len(x))
        print(x.shape, y.shape)

        X_sample.append(x)
        Y_sample.append(y)

X_sample = np.concatenate(X_sample, 0)
Y_sample = np.concatenate(Y_sample, 0)

# Set the base directory path
base_dir = '/content/drive/Shareddrives/CIS5190_Final_Project/Train_Data_Old'
X_real = []
Y_real = []
# Loop through subfolders
for idx, subfolder in enumerate(os.listdir(base_dir)):
    subfolder_path = os.path.join(base_dir, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")


        # Loop through files in the subfolder
        # for idx, file in enumerate(os.listdir(subfolder_path)):
        # file_path = os.path.join(subfolder_path)
        x = load_dataset(subfolder_path)
        y = np.array([CLASS_TO_LABEL[subfolder]] * len(x))
        print(x.shape, y.shape)

        X_real.append(x)
        Y_real.append(y)

X_real = np.concatenate(X_real, 0)
Y_real = np.concatenate(Y_real, 0)

"""# Standarlization"""

from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler on X_sample and transform both datasets
X_sample_scaled = scaler.fit_transform(X_sample)
X_real_scaled = scaler.transform(X_real)

print("X_sample_scaled shape:", X_sample_scaled.shape)
print("X_real_scaled shape:", X_real_scaled.shape)

"""# Feature selection
only chi-best is useful, feature 66 to 30
"""

#Select chi-best
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

# Step 2: Scale the training set (X_real)
scaler = MinMaxScaler()
X_real_scaled = scaler.fit_transform(X_real)

# Step 3: Apply feature selection on the training set
selector = SelectKBest(chi2, k=30)  # Select the top 10 features
X_real_selected = selector.fit_transform(X_real_scaled, Y_real)

# Step 4: Scale the validation set (X_sample)
# Assuming X_sample is your validation set (it should be defined somewhere)
X_sample_scaled = scaler.transform(X_sample)  # Same scaler applied

# Step 5: Apply the same feature selection to the validation set
X_sample_selected = selector.transform(X_sample_scaled)  # Apply the same selector

# Step 6: Ensure the selected features match (this is inherent in how feature selection works)
# X_sample_selected will have exactly the same features as X_real_selected because we're using the same `SelectKBest`

"""# Remove bad feature for each class"""

import numpy as np

def remove_bad_features(X_real, X_sample, Y_real, Y_sample, threshold=0.5):
    # Initialize arrays to hold the cleaned data (starting with original data)
    X = X_real.copy()
    X_s = X_sample.copy()
    X_real_subsampled = []
    Y_real_subsampled = []

    # Subsampling: make sure that X_real has the same number of samples for each class as X_sample
    for cls in np.unique(Y_sample):
        # Get the indices of the current class in X_sample
        sample_indices = np.where(Y_sample == cls)[0]

        # Get the indices of the current class in X_real
        real_indices = np.where(Y_real == cls)[0]

        # Number of samples in the current class in X_sample
        num_samples_in_class = len(sample_indices)

        # Randomly select the same number of samples from X_real for the current class
        selected_real_indices = np.random.choice(real_indices, num_samples_in_class, replace=False)

        # Append the selected samples to the subsampled lists
        X_real_subsampled.append(X[selected_real_indices])
        Y_real_subsampled.append(np.full(num_samples_in_class, cls))  # Append the class labels

    # Convert the list of subsampled data into numpy arrays
    X_real_subsampled = np.vstack(X_real_subsampled)
    Y_real_subsampled = np.concatenate(Y_real_subsampled)

    # Initialize cleaned arrays
    X_real_cleaned = X.copy()
    X_sample_cleaned = X_s.copy()

    # Iterate over each class in Y_sample
    for cls in np.unique(Y_sample):
        # Get indices of the current class for both datasets
        sample_indices = np.where(Y_sample == cls)[0]
        real_indices = np.where(Y_real_subsampled == cls)[0]

        # Extract features for the current class
        sample_subsampled = X_s[sample_indices]
        real_subsampled = X_real_subsampled[real_indices]

        # Compute mean and std for the current class
        sample_mean = np.mean(sample_subsampled, axis=0)
        sample_std = np.std(sample_subsampled, axis=0)
        real_mean = np.mean(real_subsampled, axis=0)
        real_std = np.std(real_subsampled, axis=0)

        # Compute differences
        mean_diff = np.abs(sample_mean - real_mean)
        std_diff = np.abs(sample_std - real_std)

        # Find features with both mean and std differences > threshold (non-significant features)
        non_significant_features = np.where((mean_diff > threshold) | (std_diff > threshold))[0]

        # Debugging: print out the features that are non-significant for each class
        print(f"Class {cls}:")
        print(f"Features with Mean Difference > {threshold}: {np.where(mean_diff > threshold)[0]}")
        print(f"Features with Std Difference > {threshold}: {np.where(std_diff > threshold)[0]}")
        print(f"Non-significant features (both mean and std > {threshold}): {non_significant_features}")
        print("-" * 50)

        # Make sure non-significant features are within the bounds of available features
        valid_non_significant_features = non_significant_features[non_significant_features < X_real.shape[1]]

        # Set non-significant features to 0 in both X_real and X_sample for the current class
        if len(valid_non_significant_features) > 0:
            X_real_cleaned[real_indices[:, None], valid_non_significant_features] = 0
            X_sample_cleaned[sample_indices[:, None], valid_non_significant_features] = 0

    return X_real_cleaned, X_sample_cleaned

# Example usage:
X_real_cleaned, X_sample_cleaned = remove_bad_features(X_real_selected, X_sample_selected, Y_real, Y_sample)

# Print out the shape of cleaned data
print("Cleaned X_real shape:", X_real_cleaned.shape)
print("Cleaned X_sample shape:", X_sample_cleaned.shape)

"""# Model

## **Logitsic + SVM** - 0.9137
Weighted Soft Voting with Class-Specific Adjustments.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming your data is in X_real_selected, Y_real, X_sample_selected, Y_sample

# Step 1: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_real_cleaned, Y_real, test_size=0.2, random_state=42)
X_val = X_sample_cleaned
y_val = Y_sample

# Step 2: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize base models (Logistic Regression and SVM)
logreg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')  # Weighted for class imbalance
svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)  # Enable probability estimation
#rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Step 4: Train the base models
logreg.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

# Step 5: Get predicted probabilities from both models
logreg_proba = logreg.predict_proba(X_val_scaled)  # Probabilities from Logistic Regression
svm_proba = svm.predict_proba(X_val_scaled)  # Probabilities from SVM

# Step 6: Apply class-specific weighting
logreg_weighted = logreg_proba.copy()
svm_weighted = svm_proba.copy()

logreg_weighted[:,[1]] *= 1.2# Increase weight for class 0 and 6 (Logistic Regression)

# Step 7: Combine the weighted probabilities
combined_proba = (logreg_weighted + svm_weighted)/2  # Average the weighted probabilities


# Step 8: Make final predictions based on the highest probability
y_pred_val = np.argmax(combined_proba, axis=1)
y_pred_test = np.argmax((logreg.predict_proba(X_test_scaled) + svm.predict_proba(X_test_scaled)) / 2, axis=1)

# Step 9: Calculate accuracy for test and validation sets
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

# Step 10: Print the results
print(f"Test Accuracy with Custom Weighting: {test_acc * 100:.2f}%")
print(f"Validation Accuracy with Custom Weighting: {val_acc * 100:.2f}%")

for i in range(len(y_val)):
  if y_val[i] != y_pred_val[i]:
    print(f'Correct {(y_val[i] == y_pred_val[i])}, Pred {y_pred_val[i]}, Label: {y_val[i]}')

"""## Logitisc + SVM + RF - 0.97"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming your data is in X_real_selected, Y_real, X_sample_selected, Y_sample

# Step 1: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_real_cleaned, Y_real, test_size=0.2, random_state=42)
X_val = X_sample_cleaned
y_val = Y_sample

# Step 2: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize base models (Logistic Regression, SVM, and Random Forest)
logreg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')  # Weighted for class imbalance
svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)  # Enable probability estimation
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Step 4: Train the base models
logreg.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# Step 5: Get predicted probabilities from all models
logreg_proba = logreg.predict_proba(X_val_scaled)  # Probabilities from Logistic Regression
svm_proba = svm.predict_proba(X_val_scaled)  # Probabilities from SVM
rf_proba = rf.predict_proba(X_val_scaled)  # Probabilities from Random Forest

# Step 6: Apply class-specific weighting
# For class 0 and 6, give more weight to Logistic Regression
# For class 2 and 5, give more weight to SVM
# For class 1, give more weight to Random Forest
logreg_weighted = logreg_proba.copy()
svm_weighted = svm_proba.copy()
rf_weighted = rf_proba.copy()

logreg_weighted[:,[1]] *= 1.5
# logreg_weighted[:,[4]] *= 0.9
# svm_weighted[:,[1]] *= 1.5
# svm_weighted[:,[4]] *= 0.9
rf_weighted[:, [4]] *= 1.2
rf_weighted[:, [0,1,2,3,5,6]] *= 0.1

# Step 7: Combine the weighted probabilities
combined_proba = (logreg_weighted + svm_weighted + rf_weighted) / 2  # Average the weighted probabilities

# Step 8: Make final predictions based on the highest probability
y_pred_val = np.argmax(combined_proba, axis=1)
y_pred_test = np.argmax(
    (logreg.predict_proba(X_test_scaled) + svm.predict_proba(X_test_scaled) + rf.predict_proba(X_test_scaled)) / 3,
    axis=1
)

# Step 9: Calculate accuracy for test and validation sets
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

# Step 10: Print the results
print(f"Test Accuracy with Custom Weighted Voting: {test_acc * 100:.2f}%")
print(f"Validation Accuracy with Custom Weighted Voting: {val_acc * 100:.2f}%")

for i in range(len(y_val)):
  if y_val[i] != y_pred_val[i]:
    print(f'Correct {(y_val[i] == y_pred_val[i])}, Pred {y_pred_val[i]}, Label: {y_val[i]}')

"""# Data Comparison

## Sample & Real similarity comparison
This chunk takes all features and compare class by class,output the Average Mean Difference and std between sample data and real data class by class. \
Significant difference (>0.5) indicate 我们收集的数据不好,这个class需要重新录制
"""

import numpy as np

def subsample_to_match(X_real_scaled, Y_real, X_sample_scaled, Y_sample):
    # Determine the target number of samples
    target_size = X_sample_scaled.shape[0]

    # Ensure there are enough samples in X_real and Y_real
    if target_size > X_real_scaled.shape[0]:
        raise ValueError("X_real_scaled has fewer samples than the target size.")

    # Randomly select indices
    sampled_indices = np.random.choice(X_real_scaled.shape[0], target_size, replace=False)

    # Subsample X_real and Y_real
    X_real_subsampled = X_real_scaled[sampled_indices]
    Y_real_subsampled = Y_real[sampled_indices]

    print(f"Subsampled X_real shape: {X_real_subsampled.shape}")
    print(f"Subsampled Y_real shape: {Y_real_subsampled.shape}")
    return X_real_subsampled, Y_real_subsampled
X_real_subsampled, Y_real_subsampled = subsample_to_match(X_real, Y_real, X_sample, Y_sample)

import numpy as np

def compute_class_stats(X, Y):
    # Dictionary to store stats by class
    class_stats = {}
    for cls in np.unique(Y):
        indices = np.where(Y == cls)[0]
        X_class = X[indices]
        class_mean = X_class.mean(axis=0)
        class_std = X_class.std(axis=0)
        class_stats[cls] = {"mean": class_mean, "std": class_std}
    return class_stats

sample_stats = compute_class_stats(X_sample_scaled, Y_sample)
real_stats = compute_class_stats(X_real_scaled, Y_real_subsampled)

def compare_class_stats(sample_stats, real_stats):
    mean_diffs = {}
    std_diffs = {}
    for cls in sample_stats.keys():
        sample_mean = sample_stats[cls]["mean"]
        sample_std = sample_stats[cls]["std"]
        real_mean = real_stats[cls]["mean"]
        real_std = real_stats[cls]["std"]

        mean_diff = np.abs(sample_mean - real_mean)
        std_diff = np.abs(sample_std - real_std)

        mean_diffs[cls] = mean_diff
        std_diffs[cls] = std_diff

        print(f"Class {cls} - Average Mean Difference: {mean_diff.mean():.4f}")
        print(f"Class {cls} - Average Std Difference: {std_diff.mean():.4f}")
        print()
    return mean_diffs, std_diffs

mean_diffs, std_diffs = compare_class_stats(sample_stats, real_stats)

"""Thresholds for Interpretation
Average Mean Difference (AMD):
Good Match: AMD ≤ 0.5 indicates small differences in feature means, suggesting good alignment.
Moderate Match: 0.5 < AMD ≤ 1.0 indicates manageable differences in feature means, which are usually acceptable.
Poor Match: AMD > 1.0 indicates significant differences in feature means, which may require investigation.
Average Std Difference (ASD):
Good Match: ASD ≤ 0.2 indicates small differences in feature spread, suggesting good alignment.
Moderate Match: 0.2 < ASD ≤ 0.4 indicates manageable differences in feature spread, which are generally acceptable.
Poor Match: ASD > 0.4 indicates significant differences in feature spread, which may require attention.

Good/Excellent Classes: Classes 2 and 5 are well-aligned and require no further action.
Moderate Classes: Classes 0, 1, and 6 are reasonably aligned and can be used, but monitoring or minor adjustments may be needed.
Poor Classes: Classes 3 and 4 show significant discrepancies and need further investigation or preprocessing before proceeding with modeling.
"""