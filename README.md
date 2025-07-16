# Disease Predictor using Machine Learning

This project predicts diseases based on symptoms using machine learning. It uses a dataset (`Training.csv`) containing symptoms as features and disease names as the target.

---

## Dataset
- **Source**: `archive.zip` (contains `Training.csv`)
- **Target Column**: `prognosis` (name of the disease)
- **Input Features**: Binary values (1 or 0) for symptom presence

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib

---

## Project Workflow
1. **Extracts Dataset**  
   Automatically extracts `archive.zip` to a `dataset/` folder.
2. **Preprocessing**  
   - Drops unnecessary columns like `Unnamed: 133`
   - Encodes disease names using `LabelEncoder`
3. **Model Training**  
   - Splits data into training and validation sets
   - Trains both:
     - `RandomForestClassifier`
     - `XGBClassifier`
4. **Evaluation**  
   - Prints accuracy of both models
   - Shows a detailed classification report for XGBoost

5. **Saving the Model**  
   - Saves the final trained XGBoost model and label encoder:
     - `disease_predictor_xgb.pkl`
     - `label_encoder.pkl`

---

## Results
| Model           | Accuracy |
|-----------------|----------|
| Random Forest   | ~96%     |
| XGBoost         | ~96%     |

XGBoost performed better and is saved as the final model.

---

## Installation

### 1. Install the required libraries:
pip install pandas scikit-learn xgboost joblib
### 2. How to run
python disease_predictor.py
### 3. Output will be samed as 
disease_predictor_xgb.pkl
label_encoder.pkl
