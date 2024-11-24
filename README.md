# Audio_project
This is a simple tryout of Audio Classification Project for CIS 5190

Class Label
Ben Franklin statue - Google Map "Ben on the Bench"
Blackboard (not white board) - Levine 4th floor bump space
Table - Levine 4th floor near window
Glass window - Levine 4th floor
Handrail - Levine 4th floor
Water fountain - Levine 4th floor
Sofa - Levine 4th floor bump space

Nov.18 -Aria
Data comparison indicate the similarity between real data collected and TA provided sample. Class 2 and 5 consider well. Class 3 and 4 have significant mean and std difference. Feature extraction (Chi-best) conducted to select the top 30 features.
Data later used to train several different model:

Simple CNN - perform fair with remove_bad T>.5 (0.77/0.46)
Complex CNN - fair with remove_bad+FS(0.99/0.49)
CNN+LSTM - fair with FS+remove_bad(0.67/0.66)
Logistic - fair with FS (0.98/0.6) particularly well on class 6,2
SVM - fair with FS (0.99/0.6) particularly well on class 0,2,5
Random forest - FS (0.96/0.42) underperform
KNN - underperform (0.95/0.42) particularly well in class 1
Logistic + SVM (0.98/0.6286)
Logistic + SVM + KNN (0.97/0.6857)
For model 1-3, CNN perform not ideally in feature classification, relatively ideal feature size with raw 66, selected 30. Issue caused by lack of data and low feature size.
Model 4-7 perform fair in test acc, each have particular outer perform in specific class. Final combined model 9.Logistic + SVM + KNN achieved 68.2% val acc, with all class 3 and 4 being labeled incorrectly and 1 class 1 incorrect. Further improvement on Class 3 and 4 data collection might result this issue.

OCA, PCA, OCA+ICA, RFE were all conducted while do not show reliable evidence.

Nov.21 -Aria
Class 3 and 4 have been re-recorded and upload for training.
Old Class 3,4 be eliminated.
Best performance FS-top 30 -> RB either >0.5
Logistic + SVM gives 91-94% val acc
Logistic + SVM + RF 97% val acc
