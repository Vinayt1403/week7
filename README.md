Week 7: Introduction to Statistics for Data Science
1. Project Overview
Project Title: Customer Churn and Sales Data Analysis
Project Goals and Objectives:
The goal of this project is to perform a comprehensive statistical analysis on customer churn and sales data. The objectives include:
•	Understanding customer behavior and identifying factors contributing to churn.
•	Analyzing sales performance across different products and regions.
•	Applying statistical methods, hypothesis testing, and regression analysis to uncover insights.
•	Providing actionable business recommendations based on data analysis.
Datasets Used:
1.	customer_churn.csv – Contains information about customer tenure, monthly charges, total charges, contract type, payment method, and churn status.
2.	sales_data.csv – Contains sales transactions including date, product, quantity, price, customer ID, region, and total sales.
2. Setup Instructions
System Requirements:
•	Python 3.10+
•	Minimum 8 GB RAM recommended
Step-by-Step Installation and Configuration:
1.	Install Python – Download from python.org and install.
2.	Set up a virtual environment (optional but recommended):
3.	python -m venv venv
4.	source venv/bin/activate  # Linux/Mac
5.	venv\Scripts\activate     # Windows
6.	Install required packages:
Save the following in requirements.txt and run:
7.	pip install -r requirements.txt


requirements.txt includes:
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
8.	Place CSV files (customer_churn.csv and sales_data.csv) in the same project directory.
9.	Run the analysis script:
10.	python analysis.py
This will generate outputs, plots, regression results, and hypothesis_tests_results.txt.

3. Code Structure
Project File Hierarchy:
Week7/
├── customer_churn.csv
├── sales_data.csv
├── analysis.py                 # Main Python script
├── requirements.txt
└── hypothesis_tests_results.txt
Code Organization:
•	Day 1: Descriptive statistics (mean, median, mode, standard deviation)
•	Day 2: Data distribution analysis (histograms, normality test)
•	Day 3: Correlation analysis (Pearson correlation and heatmap)
•	Day 4: Hypothesis testing (ANOVA, t-test, Chi-Square)
•	Day 5: Confidence intervals (95% CI calculation)
•	Day 6: Regression analysis (predicting TotalCharges)
•	Day 7: Business insights and actionable recommendations

4. Visual Documentation
Examples of Visual Outputs:
Descriptive statistics.
 

 
 
 
 
 









Distribution of churn.
 













Histogram of MonthlyCharges (Customer Data)
  









Correlation Heatmap (Customer Data)
  
Regression Plot: TotalCharges vs Tenure and MonthlyCharges
 
Distribution Total sales
 
Distribution  Total Charges
 
Distribution  Total Quantity
 
Distribution  Senior Citizens
 


5. Technical Details
Algorithms and Methods Used:
1.	Descriptive Statistics:
o	Mean, median, mode, standard deviation to summarize numeric data.
2.	Data Distribution Analysis:
o	Histograms for visual inspection of data distribution.
o	Shapiro-Wilk test for normality assessment.
3.	Correlation Analysis:
o	Pearson correlation coefficient to quantify linear relationships.
o	Heatmap visualization for easy interpretation.
4.	Hypothesis Testing:
o	ANOVA: Check if MonthlyCharges differ by Contract type.
o	t-test: Compare MonthlyCharges between senior and non-senior customers.
o	Chi-Square test: Check the association between PaperlessBilling and Churn.
5.	Confidence Intervals:
o	95% CI calculated for numeric features to estimate the range of expected values.
6.	Regression Analysis:
o	Linear regression predicting TotalCharges from Tenure and MonthlyCharges.
o	Coefficients and R-squared value used for model evaluation.

Data Structures Used:
•	Pandas DataFrames for data storage and manipulation.
•	Numpy arrays for numerical operations.
Architecture:
•	Single Python script with modular sections for each analytical task.
•	Outputs include textual summaries, plots, and hypothesis test results in a text file.

6. Testing Evidence
Test Cases and Validation:
•	Descriptive Statistics: Checked means, medians, and modes against raw data.
•	Normality Tests: Shapiro-Wilk test applied on all numeric columns.
•	Correlation Analysis: Verified correlation coefficients and plotted heatmaps.
•	Hypothesis Tests: Results saved to hypothesis_tests_results.txt and manually verified.
•	Regression Analysis: Verified predicted values and R-squared values against actual TotalCharges.

Sample Hypothesis Test Results:
Test: ANOVA: MonthlyCharges vs Contract
Statistic: 0.462
p-value: 0.641
Conclusion: No significant difference

Test: t-test: MonthlyCharges Senior vs Non-Senior
Statistic: 0.384
p-value: 0.707
Conclusion: No significant difference

Test: Chi-Square: PaperlessBilling vs Churn
Statistic: 0.000
p-value: 1.000
Conclusion: No significant association




7. Results Interpretation
•	Month-to-month customers have higher churn risk → focus retention campaigns.
•	Higher monthly charges correlate with higher total revenue.
•	Regression analysis shows both Tenure and MonthlyCharges strongly predict TotalCharges.
•	PaperlessBilling shows no significant impact on churn.
Business Recommendations:
•	Offer loyalty incentives to month-to-month customers.
•	Target high-value customers with retention campaigns.
•	Use predictive models to identify customers likely to churn early.



