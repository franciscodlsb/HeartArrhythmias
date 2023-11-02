# Work in Progress

Current files:
1. Readme.md -> now work in progress and some notes, should include the documentation for the rest of the files and general info about arrythmias
2. Folder ECG_Data -> I can't upload, should be obtained from here: https://doi.org/10.1038/s41597-020-0386-x
3. Folder Train_Test_Data -> to put created X (feat matrix) and Y (labels)
4. ECG_Features -> right now contains the whole pipeline. Should be changed to contain only the ECG function. It should be divided in the following jupyter notebooks:
    - "1. DataPre_Feature_Extraction.ipynb" - Includes data exploration, choice of preprocessing library with examples, and calls the ECG_features.py at the end to create and save the X and Y fiels in Train_Test_Data
    - "2. Arrythmia_classification.ipynb" - XGBoost, SVM, naive bayes comparison with 10 fold cross-validation. To do: 1. Organize in a presentable way, 2. maybe remove equal sampling and use imbalanced data and metrics that don't require balance (in reality most patients are healthy), 3. Optimize Naive Bayes and SVM to increase accuracy
    - "3. RNN.ipynb" - Optional, to do, only tried with FNN in pytorch, use a RNN or CNN, better for time series (https://towardsdatascience.com/time-series-classification-with-deep-learning-d238f0147d6f).


# Predicting Heart Arrythmias Using Electrocardiograms (ECG)
Heart arrhythmias, irregular heartbeats, can be a serious health concern. They can lead to complications like stroke and heart failure, and in some cases, can even be fatal. That’s where Electrocardiograms (ECG) come into play. ECGs are a powerful tool that can help us detect these irregular heartbeats early on. By analyzing the electrical activity of the heart captured in an ECG, with the help of machine learning (ML) we can automatically spot any abnormalities that might indicate a heart failure. This allows for timely medical intervention, which can prevent the arrhythmia from escalating into a more serious condition. 

In this repository I use the following recent ECG database from Nature Scientific Data to automatically diagnose, with high accuracy, the type of heart arrythmia of a patient.

Zheng, J., Zhang, J., Danioko, S. et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Sci Data 7, 48 (2020). https://doi.org/10.1038/s41597-020-0386-x


Data already denoised using: https://github.com/zheng120/ECGDenoisingTool/tree/master
including Butterworth low pass filter remove the signal with more than 50 Hz frequency. Then, local polynomial regression smoother (LOESS) to clear the effects of baseline wandering. Lastly and Non Local Means (NLM) technique to handle the remaining noise.

Image of qrs pt: https://en.wikipedia.org/wiki/QRS_complex#/media/File:SinusRhythmLabels.svg and feaures https://www.researchgate.net/publication/339385166/figure/fig6/AS:959592271994884@1605796034651/The-definition-of-height-width-and-prominence-measurements-in-this-study-The.png 
https://www.researchgate.net/figure/Typical-ECG-signal-with-its-distinctive-points-see-online-version-for-colours_fig1_324762381


Feature extraction neurokit 2.0: https://neuropsychology.github.io/NeuroKit, biosppy can also be used to do the segmentation of the ecg and then extracts the peaks with scipy.signal


# Classification levels of arrythmias
11 rythms into 4: SB, AFIB, GSVT, SR
1. SB -> Sinus bradycardia
2. AFIB -> atrial fibrillation and atrial flutter (AF)
3. GSVT ->   supraventricular tachycardia, atrial tachycardia, atrioventricular node reentrant tachycardia, atrioventricular reentrant tachycardia and sinus atrium to atrial wandering rhythm
4. SR -> sinus rhythm and sinus irregularity -> normal


# Feature Extraction
### Age
### Height
# From Diagnostics

 GE MUSE system to extract them
### Lead II (12)
# Ventricular rate in BPM
# Atrial rate in BPM
# QRS in ms
# QT interval in ms
# R axis
# T axis
# QRS count
# Q onset
# Q offset
# Mean of RR interval
# Variance of RR interval
# RR interval count

### 12 Leads (12x6*6=96)
## Qrs cOMPLEX
# Mean height Q
# Var height Q
# Mean width Q
# Var height Q
# Mean promincence Q
# Var prominence Q
# Mean height R
# Var height R
# Mean width R
# Var height R
# Mean promincence R
# Var prominence R
# Mean height S
# Var height S
# Mean width S
# Var height S
# Mean promincence S
# Var prominence S

## Non QRS
# Mean height P
# Var height P
# Mean width P
# Var height P
# Mean promincence P
# Var prominence P
# Mean height T
# Var height T
# Mean width T
# Var height T
# Mean promincence T
# Var prominence T

# Frequency bands relevant
https://www.sciencedirect.com/science/article/pii/S0898122107005019
seems to be 0-4, 4-12 and 12 to 50