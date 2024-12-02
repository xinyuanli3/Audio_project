Audio Classification Design- Xinyuan 11.30

################ Code is under construction ##############

Audio classification research based on decision trees, suitable for simple audio classification tasks (hitting objects)，small dataset (200 samples each). Currently shows good versatility and robustness.

In terms of feature extraction,
MFCC can effectively capture the acoustic features of audio.

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_25 = np.percentile(mfcc, 25, axis=1)
        mfcc_75 = np.percentile(mfcc, 75, axis=1)
        
Besides MFCC, frequency and time domain features：

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

For classification algorithms, Decision Tree (DT), Random Forest (RF), and XGBoost have achieved good results the project.

DT model performance on the test set: Accuracy: 0.4286

            'dt': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
            
RT model performance on the test set: Accuracy: 0.4857

            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            
XGBoost performance on the test set: Accuracy: 0.4571

            'xgb': XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
        
Studying of ensemble algorithms and explore other algorithms.

