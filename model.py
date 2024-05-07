import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


df = pd.read_csv('predictive_maintenance.csv')
print(df.head())
print("\n\n", df.isnull().sum())
print("\n\n", df.describe())


features_to_plot = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


target_distribution = df['Target'].value_counts(normalize=True)
target_distribution


df['Tool wear [min]'] = df['Tool wear [min]'].astype('float64')
df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype('float64')


df.rename(mapper={'Air temperature [K]': 'Air_temperature',
                    'Process temperature [K]': 'Process_temperature',
                    'Rotational speed [rpm]': 'Rotational_speed',
                    'Torque [Nm]': 'Torque',
                    'Tool wear [min]': 'Tool_wear'}, axis=1, inplace=True)

df


X = df[['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


import xgboost as xgb

model_xgb = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

cv_scores = cross_val_score(model_xgb, X_train, y_train, cv=4, scoring='accuracy')

model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)


print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


score=accuracy_score(y_test,y_pred)
score


import pickle

with open('model_pickel', 'wb') as f:
    pickle.dump(model_xgb,f)

with open('model_pickel', 'rb') as f:
    pred_model = pickle.load(f)