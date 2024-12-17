Audio Classification Design- Xinyuan 12.15

Audio classification research based on decision trees, suitable for simple audio classification tasks (hitting objects)，small dataset (200 samples each). Currently shows good versatility and robustness.

run_trained_model.py # python function for the trained model
train_model.py # code for training the model

Currently selected features and models and their corresponding accuracy are as follows:

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
        
knn performance on the test set: Accuracy:  0.4857
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski',
                p=2
            ),

SVM performance on the test set: Accuracy:   0.7429
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )

Ensemble performance on the test set: Accuracy:   0.5429
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


