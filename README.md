Disease Predictor using Machine Learning

This project is a machine learning-based disease prediction system built using the **Training.csv** dataset. It applies **Random Forest** and **XGBoost** classifiers to predict medical conditions based on symptoms.

 Dataset

* **Source**: `archive.zip` contains a CSV file named `Training.csv`.
* **Target Column**: `prognosis` (disease name)
* **Features**: Symptom presence (binary: 1/0) across multiple columns.

 Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **XGBoost**
* **Joblib**

 Project Workflow

1. **Data Extraction**
   Extracts the dataset from a `.zip` file.

2. **Preprocessing**

   * Drops unnecessary columns (e.g., `Unnamed: 133`)
   * Encodes the `prognosis` target labels using `LabelEncoder`.

3. **Model Training**

   * Splits the data into training and validation sets using `train_test_split`.
   * Trains both:

     * `RandomForestClassifier`
     * `XGBClassifier`

4. **Model Evaluation**

   * Evaluates model accuracy.
   * Prints a detailed classification report for XGBoost.

5. **Model Saving**

   * Saves the trained XGBoost model and the label encoder using `joblib`.

---

Results

| Model         | Accuracy       |
| ------------- | -------------- |
| Random Forest | \~ Your Output |
| XGBoost       | \~ Your Output |

(XGBoost showed better performance and is saved as the final model.)
 Installation

Install the required packages:

```bash
pip install xgboost scikit-learn pandas joblib
```

---
How to Run

1. Place your `archive.zip` in the project directory.
2. Run the Python script:

```bash
python disease_predictor.py
```

3. Models will be saved as:

   * `disease_predictor_xgb.pkl`
   * `label_encoder.pkl`

File Structure

```
.
├── archive.zip
├── dataset/
│   └── Training.csv
├── disease_predictor.py
├── disease_predictor_xgb.pkl
├── label_encoder.pkl
└── README.md
```
Contributing
