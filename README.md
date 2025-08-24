# 🚽 Toilify Model – Smart Restroom Event Classification  

This repository contains the **machine learning model** for the **Toilify Project**, which predicts restroom events based on sensor data such as **ammonia (NH₃), hydrogen sulphide (H₂S), and water level readings**.  

## 📌 Problem Statement  
Traditional restrooms often lack monitoring systems to detect hygiene levels and overuse. Toilify aims to make restrooms **smarter and more hygienic** by:  
- Detecting when a restroom needs cleaning  
- Identifying events like *pee detected, poop detected, flush event, or overuse*  
- Enabling better maintenance scheduling  

---

## ⚙️ Workflow  
1. **Data Collection** – Synthetic sensor data generated (NH₃, H₂S, water level, timestamps).  
2. **EDA & Preprocessing** – Cleaning, balancing classes, extracting time features (hour, weekday).  
3. **Model Training** – Multiple ML models were tested (Logistic Regression, RandomForest, LightGBM, XGBoost).  
4. **Best Model** – RandomForest was selected due to its stability and near-perfect classification.  
5. **Evaluation** – Confusion matrix, classification report, and feature importance were used for validation.  
6. **Deployment Ready** – Model saved using `joblib` for real-world use.  

---

## 📊 Model Performance  

- **Best Model:** `RandomForestClassifier`  
- **Accuracy:** ~100% (balanced across all classes)  
- **Macro F1 Score:** 0.99 – 1.00  

### 🔹 Confusion Matrix  
Shows the model correctly classifies all restroom events with almost no misclassifications.  

### 🔹 Feature Importance  
- `ammonia_ppm` → Most important (pee detection)  
- `Hydrogen_Sulphide` → Strong signal for poop detection  
- `water_level_cm` → Detects flush & overuse  
- `hour`, `weekday_encoded` → Provide context of usage  

---

## 🏷️ Event Labels  
The model predicts one of the following events:  
- **`normal`** → Restroom in normal condition  
- **`pee_detected`** → High ammonia detected  
- **`poop_detected`** → High hydrogen sulphide detected  
- **`flush_event`** → Water level spike detected  
- **`overuse`** → Continuous usage without flush  

---

## 🔮 Prediction Example  

```python
import joblib

# Load model
rf = joblib.load("randomforest_toilify.pkl")

# Mapping for labels
label_mapping = {
    0: "flush_event",
    1: "normal",
    2: "pee_detected",
    3: "poop_detected",
    4: "overuse"
}

# Prediction function
def predict_event(ammonia, h2s, water, hour, weekday):
    data = [[ammonia, h2s, water, hour, weekday]]
    pred = rf.predict(data)[0]
    return label_mapping[pred]

print(predict_event(90, 5, 28, 7, 2))  
# Output → "pee_detected"
