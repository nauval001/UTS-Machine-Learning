import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Input data
df = pd.read_csv('TravelInsurancePrediction.csv')
print("Shape dataset:", df.shape)
print(df.head())

# Preprocessing
df = df.drop_duplicates()
df['AnnualIncome'] = df['AnnualIncome'].clip(
    lower=df['AnnualIncome'].quantile(0.01),
    upper=df['AnnualIncome'].quantile(0.99))
print("Preprocessing selesai. Shape:", df.shape)

# Split data
X = df.drop('TravelInsurance', axis=1)
y = df['TravelInsurance']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Data Training:", X_train.shape[0], "| Data Testing:", X_test.shape[0])

# Transformasi
categorical_cols = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad', 'ChronicDiseases']
numerical_cols = ['Age', 'AnnualIncome', 'FamilyMembers']
# (One-Hot Encoding)
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
# (Min-Max Scaling)
scaler = MinMaxScaler()
X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])
print("Transformasi selesai.")

# Seleksi Fitur
# (SelectKBest)
selector_kbest = SelectKBest(score_func=f_classif, k=5)
X_train_kbest = selector_kbest.fit_transform(X_train_encoded, y_train)
X_test_kbest = selector_kbest.transform(X_test_encoded)
# (RFE)
estimator = DecisionTreeClassifier(random_state=42)
selector_rfe = RFE(estimator=estimator, n_features_to_select=5)
X_train_rfe = selector_rfe.fit_transform(X_train_encoded, y_train)
X_test_rfe = selector_rfe.transform(X_test_encoded)
print("Feature Selection selesai.")

# Resampling
# (Terapkan ROS)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_kbest, y_train)
# (Terapkan Tomek Links)
tomek = TomekLinks()
X_train_final, y_train_final = tomek.fit_resample(X_train_resampled, y_train_resampled)
print("Distribusi setelah Resampling:")
print(pd.Series(y_train_final).value_counts())

# Model Training
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_final, y_train_final)
print("Model berhasil dilatih.")

# Testing & Model Evaluation
y_pred = dt_model.predict(X_test_kbest)
# (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Beli', 'Beli'], yticklabels=['Tidak Beli', 'Beli'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Gambar Confusion Matrix disimpan sebagai 'confusion_matrix.png'")
# (Metrik Evaluasi)
print("=== Hasil Evaluasi ===")
print(f"Akurasi  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Presisi  : {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")