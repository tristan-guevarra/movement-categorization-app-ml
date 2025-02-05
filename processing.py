import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score, f1_score, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA




with h5py.File('./hdf5_data.h5', 'r') as hdf:
    # Access the training dataset
    train_dataset = pd.DataFrame(hdf['dataset/train/trainset'][:])
    test_dataset = pd.DataFrame(hdf['dataset/test/testset'][:])

# Pre-Processing

#print(test_dataset)
window_size = 10


# Create the preprocessed training set
x_sma = train_dataset.iloc[:, 1].rolling(window_size).mean()
y_sma = train_dataset.iloc[:, 2].rolling(window_size).mean()
z_sma = train_dataset.iloc[:, 3].rolling(window_size).mean()

# Put the training processed data into a dataframe called prep_train
prep_train = pd.DataFrame({'X (m/s^2)': x_sma, 'Y (m/s^2)': y_sma, 'Z (m/s^2)': z_sma}).dropna()

prep_train = pd.concat([prep_train, train_dataset.iloc[window_size-1:, 4].rename('Label')], axis=1)


# Create the preprocessed testing set
x_train_sma = test_dataset.iloc[:, 1].rolling(window_size).mean()
y_train_sma = test_dataset.iloc[:, 2].rolling(window_size).mean()
z_train_sma = test_dataset.iloc[:, 3].rolling(window_size).mean()


# Put the training processed data into a dataframe called prep_train
prep_test = pd.DataFrame({'X (m/s^2)': x_train_sma, 'Y (m/s^2)': y_train_sma, 'Z (m/s^2)': z_train_sma}).dropna()

prep_test = pd.concat([prep_test, test_dataset.iloc[window_size-1:, 4].rename('Label')], axis=1)



# This is just to debug / view the plot of the processed data
prep_train.to_csv('preprocessed_train_data.csv')


# Feature Extraction
def extract_features(output_df, input_df, window_size):
    for i in range(1, int(len(input_df)/500)): # How many 5 second windows there are (500hz polling rate)
        output_df.iloc[i, 30] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, 3].values[0] #Put the labels in the features data
        for j in range(3): # One for each direction (X, Y, Z)
            output_df.iloc[i, j * 10] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].mean()
            output_df.iloc[i, j * 10 + 1] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].min()
            output_df.iloc[i, j * 10 + 2] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].max()
            output_df.iloc[i, j * 10 + 3] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].std()
            output_df.iloc[i, j * 10 + 4] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].kurt()
            output_df.iloc[i, j * 10 + 5] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].skew()
            output_df.iloc[i, j * 10 + 6] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].median()
            output_df.iloc[i, j * 10 + 7] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].quantile(0.75)
            output_df.iloc[i, j * 10 + 8] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].max() - input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].min()
            output_df.iloc[i, j * 10 + 9] = input_df.iloc[i * 500 - 4:(i * 500) + 500 - window_size + 1, j].var()

    return output_df

# Create an empty list, with the correct number of 5s windows (Training Set & Testing Set)
features = pd.DataFrame(index=range(int(len(prep_train)/500)), columns=['x_mean', 'x_min', 'x_max', 'x_std', 'x_kurtosis', 'x_skew','x_median', 'x_quantile', 'x_range', 'x_variance', 'y_mean', 'y_min', 'y_max', 'y_std', 'y_kurtosis', 'y_skew','y_median', 'y_quantile', 'y_range', 'y_variance','z_mean', 'z_min', 'z_max', 'z_std', 'z_kurtosis', 'z_skew','z_median', 'z_quantile', 'z_range', 'z_variance', 'Label'])
test_features = pd.DataFrame(index=range(int(len(prep_test)/500)), columns=['x_mean', 'x_min', 'x_max', 'x_std', 'x_kurtosis', 'x_skew','x_median', 'x_quantile', 'x_range', 'x_variance', 'y_mean', 'y_min', 'y_max', 'y_std', 'y_kurtosis', 'y_skew','y_median', 'y_quantile', 'y_range', 'y_variance','z_mean', 'z_min', 'z_max', 'z_std', 'z_kurtosis', 'z_skew','z_median', 'z_quantile', 'z_range', 'z_variance', 'Label'])

features = extract_features(features, prep_train, window_size)
test_features = extract_features(test_features, prep_test, window_size)

features.to_csv('calculated-feats.csv') #save to csv for debug and for app

features_labels = features.iloc[1:, 30].tolist()
features = features.iloc[1:, :29]

test_labels = test_features.iloc[1:, 30].tolist()
test_features = test_features.iloc[1:, :29]

#print(test_labels)
#print(test_features)



#Normalize the Data
sc = StandardScaler()
features = sc.fit_transform(features.iloc[:, :])
test_features = sc.fit_transform(test_features.iloc[:, :])

#Classifier
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(l_reg)

#print(features)
#print(features_labels)

# Train the model with the extracted features
clf.fit(features, features_labels)
y_pred = clf.predict(test_features)
y_clf_prob = clf.predict_proba(test_features)



print("y_pred is:", y_pred)
print("y_clf_prob is:", y_clf_prob)

print('accuracy is', accuracy_score(test_labels, y_pred))

conf_matrix = confusion_matrix(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred, average='weighted') # Use 'binary' for binary classification
roc_auc = roc_auc_score(test_labels, y_clf_prob[:, 1]) # Assuming the second column for positive class probabilities for binary classification
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'F1 Score (weighted): {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(test_labels, y_pred)
plt.title('Confusion Matrix')
plt.show()
fpr, tpr, _ = roc_curve(test_labels, y_clf_prob[:, 1])
plt.figure(figsize=(8, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Logistic Regression').plot()
plt.title('ROC Curve')
plt.show()


# Plotting a decision boundary with PCA of 2
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
test_features_pca = pca.transform(test_features)
clf_pca = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
clf_pca.fit(features_pca, features_labels)
plt.figure(figsize=(10, 8))
display = DecisionBoundaryDisplay.from_estimator( clf_pca, features_pca, response_method="predict", cmap=plt.cm.coolwarm, alpha=0.5 )
display.ax_.scatter(features_pca[:, 0], features_pca[:, 1], c=features_labels, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
plt.title('Decision Boundary with PCA Components')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
