## Fraud Detection System

## Overview
This project develops a machine learning pipeline to detect fraudulent credit card transactions. Fraud detection is a classic **imbalanced classification problem**, where fraudulent transactions make up less than **0.2%** of all data.  
The goal was to compare baseline models, oversampling (SMOTE), and anomaly detection approaches, while evaluating with metrics that matter in imbalanced datasets — **Recall** and **PR-AUC**.  


## Dataset
- **Source:** [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size:** 284,807 transactions, 31 columns  
- **Features:** `V1–V28` (PCA-transformed), `Amount`, `Time`  
- **Target:** `Class` (0 = Non-Fraud, 1 = Fraud)  
- **Class distribution:**  
    - Non-Fraud: 284,315 (~99.8%)  
    - Fraud: 492 (~0.2%)  

## Exploartory Data Analysis (EDA)
- **No missing values** in the dataset.  
- Features are numerical (already PCA-transformed for confidentiality).  
- Severe **class imbalance** visualized with bar plots.  
- Feature scaling and resampling were critical to improve model performance.

## Models Implemented

### 1. Logistic Regression
- With and without **SMOTE oversampling**  
- Pros: High recall (caught most frauds)  
- Cons: Very low precision → many false positives 

### 2. Random Forest  
- Trained with **class weights**  
- Strong balance between precision and recall  
- Improved further when combined with SMOTE  

### 3. Isolation Forest (Anomaly Detection)  
- Unsupervised approach  
- Achieved good ROC-AUC but **very low PR-AUC**  
- Struggled to balance fraud detection in practice  

## Results

### Metrics Table
| Model                        | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|------------------------------|-----------|--------|----------|---------|--------|
| Logistic Regression          | 5.10      | 91.84  | 9.67     | 97.35   | 75.26  |
| Logistic Regression + SMOTE  | 12.48     | 89.90  | 21.92    | 97.53   | 78.58  |
| Random Forest                | 96.05     | 74.49  | 84.25    | 95.29   | 86.00  |
| Random Forest + SMOTE        | 82.65     | 82.65  | 82.65    | 96.04   | 87.53  |
| Isolation Forest             | 31.13     | 33.67  | 32.35    | 97.42   | 32.46  |

### Visualization Highlights
- **Precision:** Random Forest had the highest precision; Logistic Regression struggled.  
- **Recall:** Logistic Regression variants had very high recall, but poor precision.  
- **F1-Score:** Random Forest (with and without SMOTE) achieved the best balance.  
- **PR-AUC:** Random Forest with SMOTE performed best overall.  

## Conclusion
1. Fraud detection requires focusing on **Recall** and **PR-AUC** more than accuracy.  
2. **Logistic Regression:** High recall but too many false positives — not practical alone.  
3. **Random Forest:** Balanced precision & recall, strong F1, best PR-AUC.  
4. **Random Forest + SMOTE:** Slightly improved recall & PR-AUC → **best overall model**.  
5. **Isolation Forest:** Underperformed compared to supervised approaches.  

**Final Choice:** **Random Forest with SMOTE** is the most reliable for fraud detection in this dataset.  

## Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn  
- **Environment:** Jupyter Notebook / VS Code  

## How to Run  
1. Clone the repo:  
   ```bash
   git clone https://github.com/haydenchang/Fraud-Detection-System.git
   cd Fraud-Detection-System