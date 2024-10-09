import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.patches as mpatches
from math import pi

# Load and preprocess data
data = r"C:\Users\MONISH VAYUGANDLA\Downloads\HOSPITAL_MANAGEMENT_SYSTEM\framingham.csv"
df = pd.read_csv(data)

# Define column names
col_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 
             'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
df.columns = col_names

# Encode categorical data
df['male'] = df['male'].map({0: 'Female', 1: 'Male'})  # For stacked bar later
df_encoded = pd.get_dummies(df, drop_first=True)

# Train-test split
X = df_encoded.drop('TenYearCHD', axis=1)
y = df_encoded['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred)
print('Model accuracy score: {0:0.4f}'.format(accuracy_default))

# Feature importance plot
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Radar plot (for actual vs predicted comparison)
categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']
actual_stats = [accuracy_default, 0.75, 0.80, 0.77]  # Example values
predicted_stats = [accuracy_default, 0.78, 0.82, 0.79]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
actual_stats += actual_stats[:1]
predicted_stats += predicted_stats[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
ax.plot(angles, actual_stats, linewidth=1, linestyle='solid', label="Actual")
ax.fill(angles, actual_stats, 'b', alpha=0.1)
ax.plot(angles, predicted_stats, linewidth=1, linestyle='solid', label="Predicted")
ax.fill(angles, predicted_stats, 'r', alpha=0.1)
plt.legend(loc='upper right')
plt.title('Radar plot for Actual vs Predicted')
plt.show()

# Violin plot (updated with hue)
plt.figure(figsize=(8, 6))
sns.violinplot(x='TenYearCHD', y='age', data=df, hue='TenYearCHD', palette='muted', legend=False)
plt.title('Violin Plot of Age by TenYearCHD')
plt.show()

# Stacked bar plot for TenYearCHD and gender (male/female)
ct = pd.crosstab(df['male'], df['TenYearCHD'])
ct.plot(kind='bar', stacked=True, color=['#8da0cb', '#fc8d62'])
plt.title('Stacked Bar Plot of CHD by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='age', y='heartRate', hue='TenYearCHD', palette='coolwarm', alpha=0.7)
plt.title('Scatter Plot of Age vs Heart Rate (Colored by CHD Risk)')
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='TenYearCHD', y='glucose', data=df, palette='Set2')
plt.title('Box Plot of Glucose by TenYearCHD')
plt.show()

# Heatmap (updated for numeric columns only)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])  # Only numeric data for correlation
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Feature Correlations')
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Bar graph: Fairness graph (Model accuracy by age group)
df_test = X_test.copy()
df_test['age'] = df['age'].loc[X_test.index]
df_test['TenYearCHD'] = y_test
df_test['y_pred'] = y_pred
age_bins = [20, 30, 40, 50, 60, 70, 80]
df_test['age_group'] = pd.cut(df_test['age'], bins=age_bins)
accuracy_by_age = df_test.groupby('age_group').apply(
    lambda x: accuracy_score(x['TenYearCHD'], x['y_pred'])
)
plt.figure(figsize=(8, 6))
sns.barplot(x=accuracy_by_age.index.astype(str), y=accuracy_by_age.values, palette='coolwarm')
plt.title('Fairness Graph: Model Accuracy by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()
