# 🚽 Smart Toilet Sensor Data EDA & Processing

This repository contains exploratory data analysis (EDA), preprocessing, and feature engineering workflows on **synthetic sensor data** for smart toilet monitoring.  
The aim is to prepare clean, structured datasets for training machine learning models that can classify restroom events like `normal`, `pee_detected`, `poop_detected`, `flush_event`, and `overuse`.

---


## 📒 Notebooks

### `01_EDA_and_Data_Preprocessing.ipynb`
- Loads raw dataset and performs initial checks.  
- Parses timestamps and extracts `year`, `month`, `day`, `hour`, `minute`, `weekday`.  
- Creates labels (`event_label`) using ammonia, hydrogen sulfide, and water-level thresholds.  
- Handles imbalances by oversampling minority events and undersampling majority events.  
- Engineers additional features such as rolling averages and deltas (rate of change).  
- Exports the final processed dataset as `clean_balanced_dataset.csv`.

---

## 📊 Data

- **Raw Data (`data/raw/Toilet_data.csv.xlsx`)**  
  Synthetic dataset directly generated from Mockaroo with raw sensor values.  
