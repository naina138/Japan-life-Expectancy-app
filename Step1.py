import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Japan_life_expectancy.csv')

# Display basic info
print(df.info())
print(df.describe())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Exclude non-numeric columns
numeric_df = df.select_dtypes(include=['number'])  

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Life Expectancy Factors')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df, vars=['Life_expectancy', 'Physician', 'Income_per capita', 'Health_exp', 'Educ_exp', 'Welfare_exp'])
plt.show()
