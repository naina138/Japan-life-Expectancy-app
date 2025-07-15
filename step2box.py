import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset (if not already loaded)
df = pd.read_csv("Japan_life_expectancy.csv")  # Update with your file name

# Display summary statistics
print(df.describe())
print(df.isnull().sum())



plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop(columns=['Prefecture']))  # Excluding categorical column
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.title("Boxplot of Numeric Features")
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df['Life_expectancy'], kde=True, bins=15)
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy")
plt.ylabel("Count")
plt.show()
sns.pairplot(df[['Life_expectancy', 'Income_per capita', 'Health_exp', 'Educ_exp', 'Physician']])
plt.show()
