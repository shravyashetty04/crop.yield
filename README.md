# 🌾 Crop Yield Prediction System

> A Machine Learning-based predictive model that estimates agricultural crop output (in hg/ha) by analyzing key environmental inputs such as annual rainfall, average temperature, and pesticide usage.

---

## 📖 Project Overview

Agricultural yield is heavily dependent on dynamic environmental factors. This project aims to provide data-driven forecasting by leveraging historical data and machine learning regression algorithms. By inputting specific environmental and chemical metrics, the system helps predict the expected yield, allowing for better agricultural planning and resource management.

### 🚀 Performance & Models
* **Algorithms Implemented:** Random Forest Regressor, Decision Tree Regressor
* **Target Metric:** Crop Yield (hg/ha - hectograms per hectare)

---

## 🧰 Technology Stack

<p align="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,scikit,pandas,numpy,flask" />
  </a>
</p>

* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web Framework (If applicable):** Flask (for API/UI integration)

---

## ✨ Key Features & Methodology

### 1. Data Inputs & Feature Engineering
The model is trained on diverse agricultural datasets, extracting key predictive features:
*   ▸ **Annual Rainfall:** Impact of precipitation levels on specific crops.
*   ▸ **Average Temperature:** Climate conditions during the growing season.
*   ▸ **Pesticide Usage:** Chemical application rates (tonnes) and their correlation with yield.
*   ▸ **Item/Crop Type:** Categorical encoding for different types of agricultural products.

### 2. Model Selection & Training
The system allows for comparative analysis across multiple regression models to find the highest accuracy:
*   **Decision Tree Regressor:** Captures non-linear relationships between environmental inputs and yield.
*   **Random Forest Regressor:** An ensemble learning method that reduces overfitting and improves predictive accuracy.

### 3. Evaluation
* Models are evaluated using standard regression metrics such as **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score** to ensure reliable forecasting.

---

## 💻 Getting Started (Local Setup)

### Prerequisites
* Python 3.8+
* pip (Python package manager)

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shravyashetty04/crop-yield-prediction.git](https://github.com/shravyashetty04/crop-yield-prediction.git)
   cd crop-yield-prediction
