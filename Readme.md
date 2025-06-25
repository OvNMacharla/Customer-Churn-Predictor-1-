
# 📉 Customer Churn Predictor

A machine learning project to **predict which customers are likely to leave a telecom company** (customer churn) using classification algorithms like Logistic Regression and Decision Trees.

---

## 📌 Objective

To build an end-to-end ML pipeline that:
- Loads and preprocesses customer data
- Trains ML models (Logistic Regression, Decision Tree)
- Evaluates and improves predictions
- Helps the business retain customers

---

## 🧰 Tech Stack / Tools Used

- `Python`
- `pandas`, `numpy` – data handling
- `matplotlib` – data visualization
- `scikit-learn` – machine learning models and metrics

---

## 📁 Project Structure

```

customer-churn-predictor/
│
├── data/
│   └── customer\_churn.csv         # Dataset
│
├── src/
│   ├── preprocessing.py           # Data cleaning, encoding, feature selection
│   ├── train\_model.py             # Model training and evaluation logic
│   └── visualize.py               # Graphs and confusion matrix
│
├── main.py                        # Pipeline controller
├── requirements.txt               # All Python dependencies
└── README.md                      # You're reading this now

````

---

## ⚙️ Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

1. Ensure `customer_churn.csv` is inside the `data/` folder.

2. Run the main file:

```bash
python main.py
```

3. The terminal will show:

   * Training accuracy
   * Confusion matrix
   * Visual graphs

---

## 📊 How It Works (Behind the Scenes)

| Step                     | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| 1. **Data Load**         | Load CSV data using `pandas`                               |
| 2. **Preprocessing**     | Label encoding, missing value imputation                   |
| 3. **Split Dataset**     | `X_train`, `X_test`, `y_train`, `y_test`                   |
| 4. **Train Models**      | Logistic Regression & Decision Tree                        |
| 5. **Make Predictions**  | Predict using `X_test`                                     |
| 6. **Evaluate Accuracy** | Compare predictions (`y_pred`) with true values (`y_test`) |
| 7. **Visualize**         | Show confusion matrix with `matplotlib`                    |

---

## 📈 Example Output

```
Accuracy: 0.8162
Confusion Matrix:
 [[933 103]
  [156 217]]
```

✅ Confusion matrix and graph will also be shown in a popup window.

---

## 🔮 Future Improvements

* Add more ML models like Random Forest, XGBoost
* Deploy the model with Flask or Streamlit
* Add hyperparameter tuning
* Use cloud-hosted notebooks or APIs

---

## 🧑‍💻 Author

* Ome Venkata Nagarjuna Macharla
* [LinkedIn](https://www.linkedin.com/in/ome-venkata-nagarjuna-macharla)
* [Portfolio](https://ovn-portfolio.vercel.app/)

---