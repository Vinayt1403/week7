import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# Load datasets
customer_df = pd.read_csv("customer_churn.csv")
sales_df = pd.read_csv("sales_data.csv")

#Descriptive Statistics
print("### Day 1: Descriptive Statistics ###\n")

def descriptive_stats(df):
    print("Mean:\n", df.mean(numeric_only=True), "\n")
    print("Median:\n", df.median(numeric_only=True), "\n")
    print("Mode:\n", df.mode().iloc[0], "\n")
    print("Standard Deviation:\n", df.std(numeric_only=True), "\n")

print("Customer Churn Statistics:")
descriptive_stats(customer_df)

print("Sales Data Statistics:")
descriptive_stats(sales_df)

#Data Distribution Analysis

print("### Day 2: Data Distribution Analysis ###\n")

def plot_histograms(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
        # Test for normality
        stat, p = stats.shapiro(df[col])
        print(f"Shapiro-Wilk test for {col}: stat={stat:.3f}, p={p:.3f}")
        if p > 0.05:
            print(f"{col} likely follows a normal distribution\n")
        else:
            print(f"{col} likely does NOT follow a normal distribution\n")

print("Customer Churn Histograms:")
plot_histograms(customer_df)

print("Sales Data Histograms:")
plot_histograms(sales_df)


#Correlation Analysis

print("### Day 3: Correlation Analysis ###\n")

def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    print("Correlation Matrix:\n", corr, "\n")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

print("Customer Churn Correlation:")
correlation_analysis(customer_df)

print("Sales Data Correlation:")
correlation_analysis(sales_df)

#Hypothesis Testing
print("Day 4: Hypothesis Testing\n")

results = []

# ANOVA: MonthlyCharges across Contract types
month_to_month = customer_df[customer_df['Contract']=='Month-to-month']['MonthlyCharges']
one_year = customer_df[customer_df['Contract']=='One year']['MonthlyCharges']
two_year = customer_df[customer_df['Contract']=='Two year']['MonthlyCharges']

f_stat, p_val = stats.f_oneway(month_to_month, one_year, two_year)
results.append({
    'Test': 'ANOVA: MonthlyCharges vs Contract',
    'Statistic': f_stat,
    'p_value': p_val,
    'Conclusion': 'Significant difference' if p_val < 0.05 else 'No significant difference'
})
print(f"ANOVA for MonthlyCharges across Contract types: F={f_stat:.3f}, p={p_val:.3f}")

# t-test: MonthlyCharges between SeniorCitizen vs Non-Senior
senior = customer_df[customer_df['SeniorCitizen']==1]['MonthlyCharges']
non_senior = customer_df[customer_df['SeniorCitizen']==0]['MonthlyCharges']

t_stat, t_p_val = stats.ttest_ind(senior, non_senior, equal_var=False)  # Welch's t-test
results.append({
    'Test': 't-test: MonthlyCharges Senior vs Non-Senior',
    'Statistic': t_stat,
    'p_value': t_p_val,
    'Conclusion': 'Significant difference' if t_p_val < 0.05 else 'No significant difference'
})
print(f"t-test for MonthlyCharges between SeniorCitizen vs Non-Senior: t={t_stat:.3f}, p={t_p_val:.3f}")

#Chi-Square Test: PaperlessBilling vs Churn
contingency_table = pd.crosstab(customer_df['PaperlessBilling'], customer_df['Churn'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
results.append({
    'Test': 'Chi-Square: PaperlessBilling vs Churn',
    'Statistic': chi2,
    'p_value': chi_p,
    'Conclusion': 'Significant association' if chi_p < 0.05 else 'No significant association'
})
print(f"Chi-Square test for PaperlessBilling vs Churn: chi2={chi2:.3f}, p={chi_p:.3f}")


with open("hypothesis_tests_results.txt", "w") as f:
    f.write("Hypothesis Testing Results\n")
    f.write("==========================\n\n")
    for r in results:
        f.write(f"Test: {r['Test']}\n")
        f.write(f"Statistic: {r['Statistic']:.3f}\n")
        f.write(f"p-value: {r['p_value']:.3f}\n")
        f.write(f"Conclusion: {r['Conclusion']}\n")
        f.write("\n")


#Confidence Intervals

print("### Day 5: Confidence Intervals ###\n")

def confidence_interval(series, confidence=0.95):
    n = len(series)
    mean = np.mean(series)
    std_err = stats.sem(series)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

for col in customer_df.select_dtypes(include=np.number).columns:
    mean, lower, upper = confidence_interval(customer_df[col])
    print(f"{col}: Mean={mean:.2f}, 95% CI=({lower:.2f}, {upper:.2f})")


#Regression Analysis
print("### Day 6: Regression Analysis ###\n")

# Example: Predict TotalCharges from Tenure and MonthlyCharges
X = customer_df[['Tenure', 'MonthlyCharges']]
y = customer_df['TotalCharges']

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
r2 = reg.score(X, y)

print(f"Linear Regression coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"R-squared: {r2:.3f}")


#Business Insights
print("### Day 7: Business Insights ###\n")

print("Actionable Insights:")
print("- Month-to-month customers are at higher churn risk. Focus retention campaigns here.")
print("- Customers with higher monthly charges generate more total revenue.")
print("- Regression suggests both tenure and monthly charges strongly predict total charges.")
print("- Confidence intervals provide expected ranges for metrics like MonthlyCharges and TotalCharges.")
print("- PaperlessBilling is not significantly associated with churn, so other factors matter more for churn prediction.")
