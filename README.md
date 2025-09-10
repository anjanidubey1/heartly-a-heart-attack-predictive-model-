# ❤️ Heartly: Heart Attack Predictive Model

Heartly is a machine learning project designed to **predict the risk of heart attacks** using patient health data. Built with the **Random Forest algorithm**, this model aims to provide accurate and reliable predictions to assist in preventive healthcare.

The project includes functionalities for:

* 📂 **Data Insertion**
* 🧹 **Data Preprocessing**
* 🏋️ **Model Training**
* 🔮 **Prediction**

---

## 🚀 Features

1. **Data Insertion**

   * Load datasets in `.csv` format.
   * Accepts health-related parameters like age, gender, cholesterol, blood pressure, etc.

2. **Data Preprocessing**

   * Handles missing values.
   * Encodes categorical features.
   * Normalizes numerical data.
   * Splits data into training and testing sets.

3. **Model Training**

   * Uses the **Random Forest Classifier** for robust performance.
   * Supports hyperparameter tuning.
   * Displays accuracy, precision, recall, and F1-score.

4. **Prediction**

   * Predicts heart attack risk based on new patient input.
   * Outputs probability of risk.
   * Easy-to-use interface for quick predictions.

---

## 🛠️ Tech Stack

* **Programming Language**: Python 🐍
* **Libraries**:

  * `pandas` → Data handling
  * `numpy` → Numerical operations
  * `scikit-learn` → Machine learning (Random Forest, metrics, preprocessing)
  * `matplotlib / seaborn` → Data visualization (optional)

---

## 📊 Dataset

The model is trained on publicly available heart disease datasets (e.g., **UCI Heart Disease dataset** or Kaggle datasets).

* Features include:

  * Age
  * Gender
  * Resting blood pressure
  * Cholesterol level
  * Maximum heart rate
  * Fasting blood sugar
  * Chest pain type
  * And more…

---

## 🔧 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/heartly.git
cd heartly
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Model

```bash
python heartly.py
```

---

## 📈 Example Workflow

1. Insert patient data (manually or from a CSV).
2. Preprocess the data (cleaning, scaling, encoding).
3. Train the Random Forest model.
4. Predict whether the patient is at **risk of heart attack** or not.

**Sample Output:**

```text
Prediction: High Risk of Heart Attack
Probability: 78%
```

---

## 📂 Project Structure

```
heartly/
│
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks for EDA
├── src/                  # Source code
│   ├── preprocessing.py  # Data preprocessing functions
│   ├── model.py          # Model training & evaluation
│   ├── predict.py        # Prediction script
│
├── requirements.txt      # Dependencies
├── heartly.py            # Main execution file
└── README.md             # Project documentation
```

---

## 📌 Future Improvements

* Deploy model with a **Flask/Django web app** or **Streamlit dashboard**.
* Support for **real-time patient data**.
* Integration with **wearable health devices**.
* Enhanced model performance with deep learning.

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo.
* Create a feature branch (`git checkout -b feature-name`).
* Commit changes (`git commit -m "Added feature"`).
* Push branch (`git push origin feature-name`).
* Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* UCI Machine Learning Repository
* Kaggle Heart Disease Dataset
* Scikit-learn Documentation

