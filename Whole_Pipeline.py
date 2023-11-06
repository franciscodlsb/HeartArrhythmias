import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biosppy import utils
from biosppy import signals
import neurokit2 as nk
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Initialize Variables
sampling_rate = 500

# Info Files
data_dir = 'ECG_Data/Denoised' # ECG files
Rhythms = pd.read_excel('ECG_Data/RhythmNames.xlsx')
Diagnostics = pd.read_excel('ECG_Data/Diagnostics.xlsx')


# Labels in the Diagnostics file:
Rythm = Diagnostics['Rhythm']
    # Have to be placed in the following group
Rythms_groups = {'SR':['SR', 'SA'], # SA = sinus irregularity, bad label
                 'AFIB':['AF', 'AFIB'], 
                 'GSVT':['SVT','AT', 'AVNRT', 'AVRT', 'SAAWR', 'ST'],
                 'SB':['SB']}
reverse_groups = {val: key for key, values in Rythms_groups.items() for val in values}
    # Create y with groups (On)
y = np.array([reverse_groups.get(i, None) for i in Rythm])
#pd.value_counts(y)



# Function to create X
def ECG_Features(data_dir, sr=500):
    """**Automated feature extraction of ECG signals**

    This function loads every ECG file in the data_dir, and for every channel it performs
    the detection of the R peaks, with this it detects the P, Q, S, T peaks and valleys using
    the biosppy library, it then calculates the mean height, width and prominence for each,
    and the kurtosis, flatline_percentage, skewness and main bands power

    Parameters
    ----------
    data_dir : path where ECG files are located
    sr : sampling rate, int

    Returns
    -------
    X : DataFrame containing all the calculated features for each channel (cols) for all the patients (rows)

    """
    # Get list of files
    patients = os.listdir(data_dir)

    # Allocation of features dataframe
    channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    channel_features = [ # H = Height, P = prominence, W = Width
        # QRS Complex
        'mean_H_Q', 'mean_H_R', 'mean_H_S', 'var_H_Q', 'var_H_R', 'var_H_S',
        'mean_P_Q', 'mean_P_R', 'mean_P_S', 'var_P_Q', 'var_P_R', 'var_P_S',
        'mean_W_QRS', 'var_W_QRS',
        #Non QRS
        'mean_H_T', 'var_H_T', 'mean_W_T', 'var_W_T', 'mean_P_T', 'var_P_T',
        'mean_H_P', 'var_H_P', 'mean_W_P', 'var_W_P', 'mean_P_P', 'var_P_P',
        'mean_W_nonQRS', 'var_W_nonQRS',
        # Other
        'kurtosis', 'flatline_percentage', 'skewness','Hz0to4', 'Hz4to12', 'Hz12to30']
    combined_features = [f'{channel}_{feature}' for channel in channels 
                        for feature in channel_features]
    X = pd.DataFrame(np.nan, index=range(len(patients)), columns=combined_features)

    # Variables outputs of the biosppy
    outputs_ecg = ('ts', 'filtered', 'rpeaks', 'templates_ts',
                    'templates', 'heart_rate_ts', 'heart_rate')

    for row, patient in enumerate(patients): # iterate through every patients ecg

        # verbose
        if row % 500 == 0:
            print(f"Starting with file '{row}', filename: {patient}")

        # Load data
        ecg_channels = pd.read_csv(os.path.join(data_dir, patient), header=None, names=channels)

        for channel in channels:
            
            ecg_signal = ecg_channels[channel].to_numpy() # channel signal

            # Biossppy initial processing
            try:
                ecg_proc = utils.ReturnTuple(# Process ecg: detects R peaks and HR
                    signals.ecg.ecg(ecg_signal, sampling_rate=500, show=False, interactive=False),
                    outputs_ecg)
            except:
                continue 
            
            # Process and fill X dataframe (what's possible)
            
            ## Kurtosis
            X.loc[X.index[row], channel + '_kurtosis'] = signals.ecg.kSQI(ecg_signal)
            ## Flatline percentage: % of signal where the abs value of the dx is lower than threshold
            X.loc[X.index[row], channel + '_flatline_percentage'] = signals.ecg.pSQI(ecg_signal, f_thr=0.01)
            ## Skewness
            X.loc[X.index[row], channel + '_skewness'] = signals.ecg.sSQI(ecg_signal)

            ## Band power
            freqs, power = signals.tools.power_spectrum(ecg_signal, 500) # Power spectrum
            bands = {'Hz0to4': [0, 3.999], 'Hz4to12': [4, 11.999], 'Hz12to30': [12, 29.999]}
            for band, freq_range in bands.items():
                X.loc[X.index[row], channel + '_' + band] = signals.tools.band_power(freqs, power, freq_range).as_dict().values()


            ## Q,R,S complex

            ### R
            #### R Height
            X.loc[X.index[row], channel + '_mean_H_R'] = np.nanmean(ecg_signal[ecg_proc['rpeaks']])
            X.loc[X.index[row], channel + '_var_H_R'] = np.nanvar(ecg_signal[ecg_proc['rpeaks']])
            try:
                ### R Prominence
                R_Ps = np.nanmin(np.stack([
                    ecg_signal[ecg_proc['rpeaks']]-ecg_signal[Q_positions],
                    ecg_signal[ecg_proc['rpeaks']]-ecg_signal[S_positions] ]), axis = 0)
                X.loc[X.index[row], channel + '_mean_P_R'] = np.nanmean(R_Ps)
                X.loc[X.index[row], channel + '_var_P_R'] = np.nanvar(R_Ps)
            except:
                pass

            ### Q
            try:
                Q_positions, Q_start_positions = signals.ecg.getQPositions(ecg_proc, show=False) # Q pos
                #### Q Height
                X.loc[X.index[row], channel + '_mean_H_Q'] = np.nanmean(ecg_signal[Q_positions])
                X.loc[X.index[row], channel + '_var_H_Q'] = np.nanvar(ecg_signal[Q_positions])
                #### Q Prominence
                Q_Ps = np.nanmin(np.stack([ # substract closests countours, mix in array, calculate min of each diff
                    ecg_signal[Q_positions]-ecg_signal[ecg_proc['rpeaks']],
                    ecg_signal[Q_positions]-ecg_signal[Q_start_positions] ]), axis = 0)
                X.loc[X.index[row], channel + '_mean_P_Q'] = np.nanmean(Q_Ps)
                X.loc[X.index[row], channel + '_var_P_Q'] = np.nanvar(Q_Ps)            
            except:
                pass

            ### S
            try: 
                S_positions, S_end_positions = signals.ecg.getSPositions(ecg_proc, show=False) # S pos
                #### S Height
                X.loc[X.index[row], channel + '_mean_H_S'] = np.nanmean(ecg_signal[S_positions])
                X.loc[X.index[row], channel + '_var_H_S'] = np.nanvar(ecg_signal[S_positions])
                #### S Prominence
                S_Ps = np.nanmin(np.stack([
                    ecg_signal[S_positions]-ecg_signal[ecg_proc['rpeaks']],
                    ecg_signal[S_positions]-ecg_signal[S_end_positions] ]), axis = 0)
                X.loc[X.index[row], channel + '_mean_P_S'] = np.nanmean(S_Ps)
                X.loc[X.index[row], channel + '_var_P_S'] = np.nanvar(S_Ps)                
            except:
                pass

            ### Mean_W_QRS and var_W_QRS in ms (1/500*1000)
            try:
                X.loc[X.index[row], channel + '_mean_W_QRS'] = np.nanmean((
                    ecg_signal[S_end_positions] - ecg_signal[Q_start_positions])*2)
                X.loc[X.index[row], channel + '_var_W_QRS'] = np.nanvar((
                    ecg_signal[S_end_positions] - ecg_signal[Q_start_positions])*2)
            except:
                pass


            ## Non-QRS

            ### T
            try:
                T_positions, T_start_positions, T_end_positions = signals.ecg.getTPositions(ecg_proc, show=False) # T pos
                #### T Height
                X.loc[X.index[row], channel + '_mean_H_T'] = np.nanmean(ecg_signal[T_positions])
                X.loc[X.index[row], channel + '_var_H_T'] = np.nanvar(ecg_signal[T_positions])
                #### T Width (ms)
                X.loc[X.index[row], channel + '_mean_W_T'] = np.nanmean(
                    (ecg_signal[T_end_positions] - ecg_signal[T_start_positions])*2)
                X.loc[X.index[row], channel + '_var_W_T'] = np.nanvar(
                    (ecg_signal[T_end_positions] - ecg_signal[T_start_positions])*2)
                #### T Prominence
                T_Ps = np.nanmin(np.stack([ 
                    ecg_signal[T_positions]-ecg_signal[T_end_positions],
                    ecg_signal[T_positions]-ecg_signal[T_start_positions] ]), axis = 0)
                X.loc[X.index[row], channel + '_mean_P_T'] = np.nanmean(T_Ps)
                X.loc[X.index[row], channel + '_var_P_T'] = np.nanvar(T_Ps)

            except:
                pass

            ### P
            try:
                P_positions, P_start_positions, P_end_positions = signals.ecg.getPPositions(ecg_proc, show=False) # T pos
                #### P Height
                X.loc[X.index[row], channel + '_mean_H_P'] = np.nanmean(ecg_signal[P_positions])
                X.loc[X.index[row], channel + '_var_H_P'] = np.nanvar(ecg_signal[P_positions])
                #### P Width (ms)
                X.loc[X.index[row], channel + '_mean_W_P'] = np.nanmean(
                    (ecg_signal[P_end_positions] - ecg_signal[P_start_positions])*2)
                X.loc[X.index[row], channel + '_var_W_P'] = np.nanvar(
                    (ecg_signal[P_end_positions] - ecg_signal[P_start_positions])*2)
                #### P Prominence
                P_Ps = np.nanmin(np.stack([ 
                    ecg_signal[P_positions]-ecg_signal[P_end_positions],
                    ecg_signal[P_positions]-ecg_signal[P_start_positions] ]), axis = 0)
                X.loc[X.index[row], channel + '_mean_P_P'] = np.nanmean(P_Ps)
                X.loc[X.index[row], channel + '_var_P_P'] = np.nanvar(P_Ps)
            except:
                pass
            
            ## Non-QRS W
            try:
                nonQRS_intervals = np.array([S_end_positions[i] - Q_start_positions[i+1]
                    for i in range(len(T_end_positions)-1)])*2
                X.loc[X.index[row], channel + '_mean_W_nonQRS'] = np.nanmean(nonQRS_intervals)
                X.loc[X.index[row], channel + '_var_W_nonQRS'] = np.nanvar(nonQRS_intervals)
            except:
                pass
    
    return X


X = ECG_Features(data_dir, sr=500)
X.to_csv('ECG_Data/X.csv')


Z =  pd.concat([Diagnostics.drop(columns=['Rhythm', 'Beat']), X], axis=1)#.drop_duplicates().reset_index(drop=True)
Z.to_csv('ECG_Data/Z.csv')

a = X.isna().sum()
a.to_csv('ECG_Data/a.csv')
b = X.isna().sum(axis=1)
b.to_csv('ECG_Data/b.csv')

#Filter features with more than 1k missing values in columns
X = Z.dropna(axis=1, thresh=Z.shape[0]-1000)
# Filter rows with more than 20 missing values
X = X.dropna(axis=0, thresh=X.shape[1]-5)
# Delete the rest of the features with nans
X = X.dropna(axis=1)

y = np.array([reverse_groups.get(i, None) for i in Rythm])

y = y[X.index]



# Load your data
# X = pd.read_csv('your_data.csv')
# y = pd.read_csv('your_labels.csv')

# Drop the first column
X = X.iloc[:, 1:]

# Convert 'Gender' to binary (assuming 'male' is 1 and 'female' is 0)
X['Gender'] = X['Gender'].map({'MALE': 1, 'FEMALE': 0})

# Keep a copy of the column names
feature_names = X.columns

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Initialize classifiers
classifiers = {
    "SVM": SVC(probability=True),
    "XGBoost": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB()
}


# Initialize StratifiedKFold for 10-fold cross-validation
skf = StratifiedKFold(n_splits=10)

results = {}

for name, clf in classifiers.items():
    accuracies = []
    f1_scores = []
    tps = []
    tns = []
    rocs = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Balance classes by undersampling the majority class in the training set
        num_samples = np.bincount(y_train).min()
        X_train_balanced, y_train_balanced = [], []

        for i in np.unique(y_train):
            idx = np.where(y_train == i)[0]
            np.random.shuffle(idx)
            X_train_balanced.append(pd.DataFrame(X_train).iloc[idx[:num_samples]])
            y_train_balanced.append(y_train[idx[:num_samples]])

        X_train_balanced = pd.concat(X_train_balanced)
        y_train_balanced = np.concatenate(y_train_balanced)

        # Train classifier
        clf.fit(X_train_balanced, y_train_balanced)

        # Make predictions (ignoring rows with NaN values in the test set)
        isnan_rows_test = np.any(np.isnan(X_test), axis=1)
        y_pred_proba = clf.predict_proba(X_test[~isnan_rows_test])
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracies.append(accuracy_score(y_test[~isnan_rows_test], y_pred))
        f1_scores.append(f1_score(y_test[~isnan_rows_test], y_pred, average='weighted'))
        
        cm = confusion_matrix(y_test[~isnan_rows_test], y_pred)
        tp_rate = np.diag(cm) / np.sum(cm, axis=1)
        tn_rate = (np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)) / (np.sum(cm) - np.sum(cm, axis=0))
        
        tps.append(np.mean(tp_rate))
        tns.append(np.mean(tn_rate))

        # Compute ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(np.unique(y))):
            fpr[i], tpr[i], _ = roc_curve(y_test[~isnan_rows_test], y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        rocs.append(roc_auc)

    print(f"{name} Results:")
    print(f"Accuracy: {np.mean(accuracies)}")
    print(f"F1 Score: {np.mean(f1_scores)}")
    print(f"True Positives Rate: {np.mean(tps)}")
    print(f"True Negatives Rate: {np.mean(tns)}")

    # Feature importance (permutation importance used for SVM and Naive Bayes)
    perm_importance = permutation_importance(clf, X_test[~isnan_rows_test], y_test[~isnan_rows_test])
    importance = pd.DataFrame({'feature': feature_names, 'importance': perm_importance.importances_mean})
    print(importance.sort_values('importance', ascending=False))

    # Store results for plotting
    results[name] = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'True Positives Rate': np.mean(tps),
        'True Negatives Rate': np.mean(tns),
        'ROC AUC': np.mean([roc_auc[i] for i in roc_auc])
    }

# Plotting metrics to compare models
metrics = ['Accuracy', 'F1 Score', 'True Positives Rate', 'True Negatives Rate', 'ROC AUC']
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i, metric in enumerate(metrics):
    ax = axs[i//2, i%2]
    ax.bar(results.keys(), [results[name][metric] for name in results.keys()])
    ax.set_title(metric)

plt.tight_layout()
plt.show()





# FFNN
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

# Load your data
# X = pd.read_csv('your_data.csv')
# y = pd.read_csv('your_labels.csv')

# Drop the first column
X = X.iloc[:, 1:]

# Convert 'Gender' to binary (assuming 'MALE' is 1 and 'FEMALE' is 0)
X['Gender'] = X['Gender'].map({'MALE': 1, 'FEMALE': 0})

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
y = (y == 'AFIB').astype(int)  # Assuming 'AFIB' is the positive class

# Define dataset
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define model
class FFNN(nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)  # Assuming there are 4 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Initialize StratifiedKFold for 10-fold cross-validation
skf = StratifiedKFold(n_splits=10)

# Initialize lists to store metrics for each fold
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
roc_aucs = []
roc_curves = []
feature_importances = []


for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create data loaders
    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=32)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=32)

    # Initialize model
    model = FFNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train model
    for epoch in range(150):  # Assuming 100 epochs is enough
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model
    y_pred_proba = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_pred_proba.extend(outputs.numpy())


    # Calculate metrics for this fold and append to lists
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    tp_rates.append(np.mean(tp_rate))
    tn_rates.append(np.mean(tn_rate))
    roc_aucs.append(roc_auc)

    # Calculate ROC curve for this fold and append to list
    fpr, tpr, _ = roc_curve(y_test.numpy(), np.array(y_pred_proba)[:, 1])
    roc_curves.append((fpr, tpr))
    
    # Calculate feature importance for this fold and append to list
    feature_importance = torch.abs(model.fc1.weight).mean(dim=0).detach().numpy()
    feature_importances.append(feature_importance)



# Calculate average metrics across all folds
print(f"Average Accuracy: {np.mean(accuracies)}")
print(f"Average F1 Score: {np.mean(f1_scores)}")
print(f"Average True Positives Rate: {np.mean(tp_rates)}")
print(f"Average True Negatives Rate: {np.mean(tn_rates)}")
print(f"Average ROC AUC: {np.mean(roc_aucs)}")

# Calculate average ROC curve
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
# Calculate AUC for average ROC curve
mean_auc = auc(mean_fpr, mean_tpr)

# Plot average ROC curve
plt.figure()
plt.plot(mean_fpr, mean_tpr, label='Average ROC (AUC = %0.2f)' % mean_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate average feature importance across all folds
mean_feature_importance = np.mean(feature_importances, axis=0)

print(pd.DataFrame({'feature': feature_names.tolist(), 'importance': mean_feature_importance}).sort_values('importance', ascending=False))
# The method used here (weights of the first layer) is a simplification