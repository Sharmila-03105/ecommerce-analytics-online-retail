# E-commerce Customer & Revenue Analytics

An end-to-end data analytics project analyzing 1M+ real transactions from a UK-based online retailer (UCI Online Retail II). 

## 🎯 Business Problem
The business lacked visibility into customer retention, lifetime value, and revenue forecasting. This project builds a reproducible pipeline, predictive models, and an executive dashboard to guide retention and marketing strategies.

## 🛠️ Tech Stack
- **Data Engineering:** Python (Pandas, PyArrow), DuckDB (Star Schema)
- **Machine Learning:** Scikit-Learn, XGBoost, Lifelines (Survival Analysis)
- **Business Intelligence:** Power BI (DAX, Data Modeling, Interactive Dashboard)
- **Analytics:** RFM Segmentation, Cohort Retention, BG/NBD CLV Modeling, Time-Series Forecasting

## 📊 Dashboard Preview
<img width="627" height="350" alt="image" src="https://github.com/user-attachments/assets/c35b715f-a708-402b-ba6c-5806f8d4995a" />
<img width="625" height="352" alt="image" src="https://github.com/user-attachments/assets/b0e3651f-4729-48c7-8221-6354bd70eb2c" />
<img width="628" height="346" alt="image" src="https://github.com/user-attachments/assets/f82cbdd7-6634-4b57-8d80-6ad167eded52" />
<img width="620" height="352" alt="image" src="https://github.com/user-attachments/assets/e2309137-b728-4e49-8ee3-a89b7f5301e7" />


## 💡 Key Insights & Business Impact
- **Total Revenue Analyzed:** £20.05M across 40,000 orders.
- **Customer Value:** The top tier of customers (Champions) drive the vast majority of revenue, with top individuals spending over £6M.
- **Churn Risk:** Identified that 59% of registered customers are at High Risk of churning, representing £2.2M in historical revenue at stake. 
- **Forecasting:** Built a 6-month revenue forecast, highlighting the critical Q4 holiday seasonality spike (peaking at £2.8M in November).

## ⚙️ Project Architecture (The Pipeline)
1. **Raw:** Immutable Excel files (`data/raw/`).
2. **Staging:** Cleaned, flagged, and typed data using 10 documented rules (`data/staging/`).
3. **Warehouse:** DuckDB Star Schema (`fact_sales`, `dim_customer`, `dim_product`, `dim_date`).
4. **Analytics:** Python scripts for RFM, Cohorts, CLV, Churn (Walk-forward validated), and Forecasting.
5. **BI:** Power BI dashboard consuming exported CSVs from the warehouse.

## 🚀 How to Reproduce
1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Download the dataset from UCI and place it in `data/raw/`.
4. Run the pipeline scripts in order:
   - `python src/01_load_raw.py` 
   - `python src/02_stage.py`
   - `python src/03_warehouse.py`
   - *(continue through the 12_export_powerbi.py script)*

## ⚠️ Limitations & Future Work
- **No Cost Data:** Margin analysis was approximated via sensitivity tables since the dataset lacks a cost column.
- **Guest Customers:** Excluded from RFM and Churn models (but kept for revenue tracking).
- **Future Work:** Implement a FastAPI endpoint for real-time churn scoring.




