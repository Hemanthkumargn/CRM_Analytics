
# 🧠 CRM Analytics: Customer Segmentation & CLV Prediction

This repository presents a full-scale Customer Relationship Management (CRM) analytics pipeline focused on customer segmentation and lifetime value prediction using historical e-commerce data.

## 📈 Project Overview

The project analyzes an online retail dataset containing over 500,000 transactions to:

- Identify distinct customer segments using **RFM (Recency, Frequency, Monetary)** analysis.
- Predict **Customer Lifetime Value (CLV)** using **BG/NBD** and **Gamma-Gamma** probabilistic models.
- Provide actionable insights for customer retention and marketing strategy.

## 🔍 Key Features

- 📊 **RFM-Based Segmentation**: Classifies customers into segments like "Champions", "At Risk", "Loyal", and "Hibernating".
- 🔁 **CLV Prediction**: Uses `lifetimes` library to forecast purchase behavior and monetary value.
- ⏳ **1, 6, and 12-Month CLV Forecasts**: Helps with long-term business planning.
- 🌐 **Data Visualizations**: Bar plots, treemaps, and segmentation summaries for clear insights.
- 💡 **Key Finding**: Customers in Segment A show a cumulative CLV of ~$400K and average expected profit of ~$690 per transaction.

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Squarify
- Lifetimes (BG/NBD & Gamma-Gamma models)
- SQLAlchemy
- Scikit-learn (MinMaxScaler)

## 📁 Dataset

The dataset used is a historical online retail dataset for the years 2010–2011. It contains fields like Invoice ID, Product Description, Quantity, Price, Customer ID, and Invoice Date.

> **Note**: The dataset is not included in the repository due to size/privacy constraints. You may request it or use a similar public dataset for testing.

## 📌 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/crm-analytics.git
cd crm-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook or script:
```bash
jupyter notebook CRM_Analytics.ipynb
# or
python CRM Analytics.py
```

## 🤝 Contributing

Feel free to fork the repository and submit pull requests for improvements or new features!

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

### 🚀 Developed by [Your Name]

For questions or collaborations, feel free to open an issue or connect via [LinkedIn](https://linkedin.com/in/your-profile)
