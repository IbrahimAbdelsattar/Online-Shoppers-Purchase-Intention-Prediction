# 🛍️ Online Shoppers Purchase Intention Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Model-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📘 Overview

This project predicts **whether an online shopper will make a purchase (Revenue = True)** based on their browsing behavior, engagement, and session features.  
The goal is to help e-commerce businesses understand **user purchase intent** and **improve conversion strategies** using machine learning.

---

## 🚀 Project Workflow

### 1️⃣ Data Understanding & Preprocessing
- Imported and explored the **Online Shoppers Purchasing Intention dataset**.  
- Handled missing data and encoded categorical features (`Month`, `VisitorType`, `Weekend`).  
- Scaled numerical features using `StandardScaler` to normalize value ranges.  
- Detected class imbalance and applied **SMOTE** to balance the dataset.

### 2️⃣ Exploratory Data Analysis (EDA)
- Investigated correlations and visualized trends using:
  - Heatmaps for feature correlation
  - Bar and count plots for categorical insights
  - Distribution plots for session-based features  
- Identified that features like `PageValues`, `ExitRates`, and `VisitorType` had significant impact on purchase decisions.

### 3️⃣ Model Building
Trained and evaluated several machine learning models:

| Model | Technique | Accuracy | Precision (Class 0.0) | Recall (Class 0.0) | F1 (Class 0.0) | Precision (Class 1.0) | Recall (Class 1.0) | F1 (Class 1.0) |
|:--------------------|:-----------------------------------------|-----------:|------------------------:|---------------------:|-----------------------:|------------------------:|---------------------:|-----------------------:|
| Gradient Boosting | With SMOTE | 0.8927 | 0.9477 | 0.9237 | 0.9356 | 0.6382 | 0.7251 | 0.6789 |
| Gradient Boosting | No SMOTE | 0.9045 | 0.9339 | 0.9543 | 0.9440 | 0.7211 | 0.6361 | 0.6759 |
| Random Forest | With SMOTE | 0.8816 | 0.9506 | 0.9068 | 0.9282 | 0.5975 | 0.7461 | 0.6636 |
| XGBoost | No SMOTE | 0.8849 | 0.9445 | 0.9174 | 0.9308 | 0.6145 | 0.7094 | 0.6586 |
| Logistic Regression | No SMOTE | 0.8726 | 0.9533 | 0.8927 | 0.9220 | 0.5692 | 0.7644 | 0.6525 |
| Deep Neural Network | DNN (with SMOTE) | 0.8746 | 0.9488 | 0.9000 | 0.9237 | 0.5779 | 0.7382 | 0.6483 |
| Logistic Regression | With SMOTE | 0.8714 | 0.9514 | 0.8932 | 0.9213 | 0.5669 | 0.7539 | 0.6472 |
| XGBoost | With SMOTE | 0.8632 | 0.9580 | 0.8762 | 0.9153 | 0.5430 | 0.7932 | 0.6447 |
| SVM | No SMOTE | 0.8664 | 0.9506 | 0.8878 | 0.9181 | 0.5541 | 0.7513 | 0.6378 |
| SVM | With SMOTE | 0.8705 | 0.9449 | 0.8990 | 0.9214 | 0.5685 | 0.7173 | 0.6343 |
| Random Forest | No SMOTE | 0.8996 | 0.9207 | 0.9641 | 0.9419 | 0.7404 | 0.5524 | 0.6327 |
| KNN | With SMOTE | 0.7767 | 0.9351 | 0.7902 | 0.8565 | 0.3837 | 0.7042 | 0.4968 |
| KNN | No SMOTE | 0.8705 | 0.8927 | 0.9621 | 0.9261 | 0.6486 | 0.3770 | 0.4768 |

### 🏆 Best Model: **Gradient Boosting Classifier (No SMOTE)**
It achieved the highest accuracy and balanced precision-recall scores, making it the most reliable model for deployment.

---

## 💾 Model Saving

The trained model was serialized using **Joblib** for future deployment:

  💻 Deployment with Streamlit
🌐 App Overview

The Streamlit app allows users to input session-related details and instantly receive a purchase intention prediction.
It includes:

- Modern, clean, and responsive UI

- Input validation

- Real-time prediction results

- Probability-based output messages

# 1️⃣ Clone the repository
git clone https://github.com/your-username/online-shoppers-purchase-intention.git
cd online-shoppers-purchase-intention

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run app.py

📦 Online_Shopper_Purchase_Prediction
│
├── data/
│   └── online_shoppers_intention.csv
│
├── notebooks/
│   └── eda_and_modeling.ipynb
│
├── app.py                    # Streamlit deployment script
├── best_gradient_boosting_model.pkl
├── requirements.txt
└── README.md



---

Would you like me to add a **"Live Demo" badge** and a section that links to your **Streamlit deployed app** once you upload it online?  
That’ll make your README look even more professional and portfolio-ready.
