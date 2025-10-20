import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If CSV available:
df = pd.read_csv("notebooks/cleaned_pulsebat.csv")  # or correct path

print(df.head())

#Soh Dist

plt.figure(figsize=(8,5))
plt.hist(df['SOH'], bins=30, color='skyblue', edgecolor='black')
plt.title('SOH Distribution')
plt.xlabel('SOH')
plt.ylabel('Frequency')
plt.savefig('soh_histogram.png')
plt.show()

#Box Plots

plt.figure(figsize=(15,6))
df.boxplot(column=[f'U{i}' for i in range(1,22)])
plt.title('Boxplots of U1–U21 Features')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.savefig('u1_u21_boxplots.png')
plt.show()

#Correlation Block

# Select only numeric columns for correlation
numeric_cols = [f'U{i}' for i in range(1,22)] + ['SOH']
corr = df[numeric_cols].corr()

plt.figure(figsize=(18,14))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (U1–U21 & SOH)')
plt.savefig('correlation_heatmap.png')
plt.show()

