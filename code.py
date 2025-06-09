import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

zip_path = r"C:\Users\rames\OneDrive\AD\mlmini\archive.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")

df = pd.read_csv("dataset/Training.csv")
if 'Unnamed: 133' in df.columns:
    df.drop(columns=['Unnamed: 133'], inplace=True)

le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

X = df.drop(columns=['prognosis'])
y = df['prognosis']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

rf_pred = rf.predict(X_val)
xgb_pred = xgb.predict(X_val)

print("Random Forest Accuracy:", accuracy_score(y_val, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_val, xgb_pred))

print("\nXGBoost Classification Report:\n", classification_report(y_val, xgb_pred, target_names=le.classes_))
joblib.dump(xgb, 'disease_predictor_xgb.pkl')
joblib.dump(le, 'label_encoder.pkl')
