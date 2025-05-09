
# Customer Segmentation & Lifetime Value Prediction Using CRM Analytics

## Objective

The goal of this project is to develop a robust customer segmentation strategy and estimate Customer Lifetime Value (CLV) using historical retail transaction data. By analyzing customer behavior with RFM metrics and predictive modeling, we aim to uncover insights that support marketing decisions and long-term business planning.

## Dataset Overview

The dataset used in this project is an online retail dataset consisting of over 500,000 transactions from a UK-based e-commerce store between 2010 and 2011. It includes transaction details such as:

- Invoice number
- Product description
- Quantity
- Price
- Customer ID
- Invoice date

## Features Used

- InvoiceDate – Date and time of the transaction  
- Customer ID – Unique identifier for each customer  
- Quantity – Number of units purchased  
- Price – Price per unit of the product  
- TotalPrice – Computed as Quantity × Price  
- RFM Metrics –  
  - Recency: Days since last purchase  
  - Frequency: Number of transactions  
  - Monetary: Total spending

## Methodology

The project follows a structured analytics pipeline:

### 1. Data Preparation
- Removed cancelled orders and missing values
- Handled outliers in quantity and price
- Created TotalPrice and converted dates appropriately

### 2. Customer Segmentation (RFM Analysis)
- Calculated Recency, Frequency, and Monetary scores
- Created composite RFM scores and labeled customer segments (e.g., Champions, At Risk, Loyal)
- Visualized customer segments using bar plots and treemaps

### 3. Customer Lifetime Value (CLV) Modeling
- Applied BG/NBD model to estimate purchase frequency
- Used Gamma-Gamma model to estimate average profit per customer
- Predicted CLV over 1, 6, and 12 months
- Normalized and segmented customers by CLV (Segments A to D)

## Summary of Findings

- Segment A customers showed the highest lifetime value (~$400,000 cumulative CLV)
- Average expected profit for top-tier customers was approximately $690 per transaction
- RFM and CLV modeling effectively highlighted high-value and at-risk customer groups
- Treemap and bar visualizations provided actionable insights into customer composition

## Tools & Libraries

- Python  
- Pandas, NumPy – Data manipulation  
- Seaborn, Matplotlib, Squarify – Visualization  
- Lifetimes – CLV prediction using BG/NBD and Gamma-Gamma models  
- Scikit-learn – Normalization (MinMaxScaler)

## Business Value

This analytics framework enables businesses to:
- Tailor retention and loyalty campaigns to high-value customers
- Optimize resource allocation for marketing
- Predict revenue from different customer segments
