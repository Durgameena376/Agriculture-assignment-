# ============================================================
# Agriculture Yield Dataset - EDA & Machine Learning
# Week 3 Assignment
# ============================================================

# ── For Google Colab: uncomment these 2 lines ──
# from google.colab import files
# uploaded = files.upload()  # upload agriculture_yield_dataset.csv

# ── For IDLE: keep as is (CSV must be in same folder) ──

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PART A: Understanding the Dataset
# ─────────────────────────────────────────────

df = pd.read_csv("agriculture_yield_dataset.csv")

print("=" * 60)
print("PART A: UNDERSTANDING THE DATASET")
print("=" * 60)

# Q1. Dataset Overview
print("\nQ1. Dataset Overview:")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Column Names: {list(df.columns)}")
print("\n  First 10 Records:")
print(df.head(10))

# Q2. Data Types and Missing Values
print("\nQ2. Data Types:")
print(df.dtypes)
print(f"\n  Missing Values per Column:")
print(df.isnull().sum())
print(f"  Total Missing: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() == 0:
    print("  → No missing values found!")
else:
    print(f"  → Affected columns: {list(df.columns[df.isnull().any()])}")

# Q3. Descriptive Statistics
print("\nQ3. Descriptive Statistics:")
stats = df.describe()
print(stats)
num_cols = df.select_dtypes(include=np.number).columns
print(f"\n  Feature with Highest Mean    : {df[num_cols].mean().idxmax()} ({df[num_cols].mean().max():.2f})")
print(f"  Feature with Highest Std Dev : {df[num_cols].std().idxmax()} ({df[num_cols].std().max():.2f})")

# ─────────────────────────────────────────────
# PART B: Exploratory Data Analysis
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART B: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Q4. Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Q4. Distribution of Key Features", fontsize=14)

axes[0,0].hist(df['rainfall_mm'], bins=20, color='steelblue', edgecolor='black')
axes[0,0].set_title("Rainfall (mm)")
axes[0,0].set_xlabel("Rainfall (mm)")

axes[0,1].hist(df['temperature_c'], bins=20, color='tomato', edgecolor='black')
axes[0,1].set_title("Temperature (°C)")
axes[0,1].set_xlabel("Temperature (°C)")

axes[1,0].hist(df['fertilizer_kg'], bins=20, color='mediumseagreen', edgecolor='black')
axes[1,0].set_title("Fertilizer (kg)")
axes[1,0].set_xlabel("Fertilizer (kg)")

axes[1,1].hist(df['yield_ton_per_hectare'], bins=20, color='goldenrod', edgecolor='black')
axes[1,1].set_title("Yield (ton/hectare)")
axes[1,1].set_xlabel("Yield (ton/hectare)")

plt.tight_layout()
plt.savefig("q4_histograms.png")
plt.show()

print("\nQ4. Histogram Observations:")
print("""
  rainfall_mm:
    - Distribution is fairly spread across 400–1000 mm range
    - Roughly uniform/flat distribution, no strong peak
    - No extreme outliers visible

  temperature_c:
    - Values range mostly between 15–40°C
    - Slightly right-skewed with more moderate temperatures
    - Consistent spread indicating diverse climate data

  fertilizer_kg:
    - Fairly uniform distribution across the range
    - Most values between 50–300 kg
    - No strong clustering at any particular value

  yield_ton_per_hectare:
    - Slightly right-skewed distribution
    - Most yields fall between 2–7 tons per hectare
    - A few high-yield outliers visible on the right tail
""")

# Q5. Crop Type Analysis
print("\nQ5. Crop Type Analysis:")
crop_counts = df['crop_type'].value_counts()
print(crop_counts)
print(f"  → Most frequent crop: {crop_counts.idxmax()} ({crop_counts.max()} records)")

plt.figure(figsize=(8, 5))
crop_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Q5. Count Plot - Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("q5_crop_type.png")
plt.show()

# Q6. Soil Type Analysis
print("\nQ6. Soil Type Analysis:")
soil_counts = df['soil_type'].value_counts()
print(soil_counts)
print(f"  → Most common soil type: {soil_counts.idxmax()} ({soil_counts.max()} records)")

plt.figure(figsize=(8, 5))
soil_counts.plot(kind='bar', color='saddlebrown', edgecolor='black')
plt.title("Q6. Count Plot - Soil Type")
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("q6_soil_type.png")
plt.show()

# Q7. Yield Distribution
print("\nQ7. Yield Distribution:")
plt.figure(figsize=(8, 5))
plt.hist(df['yield_ton_per_hectare'], bins=25, color='goldenrod', edgecolor='black')
plt.title("Q7. Yield Distribution (ton/hectare)")
plt.xlabel("Yield (ton/hectare)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("q7_yield_dist.png")
plt.show()

skew = df['yield_ton_per_hectare'].skew()
print(f"  Skewness: {skew:.3f}")
print(f"  → {'Approximately normal (slight skew)' if abs(skew) < 0.5 else 'Slightly skewed, not perfectly normal'}")
print(f"  → Outliers: {'Present on the higher end' if skew > 0.3 else 'Not significant'}")

# Q8. Scatter Plot Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Q8. Scatter Plots vs Yield", fontsize=13)

axes[0].scatter(df['rainfall_mm'], df['yield_ton_per_hectare'], alpha=0.4, color='steelblue')
axes[0].set_xlabel("Rainfall (mm)")
axes[0].set_ylabel("Yield (ton/hectare)")
axes[0].set_title("Rainfall vs Yield")

axes[1].scatter(df['fertilizer_kg'], df['yield_ton_per_hectare'], alpha=0.4, color='mediumseagreen')
axes[1].set_xlabel("Fertilizer (kg)")
axes[1].set_ylabel("Yield (ton/hectare)")
axes[1].set_title("Fertilizer vs Yield")

plt.tight_layout()
plt.savefig("q8_scatter.png")
plt.show()

corr_rain  = df['rainfall_mm'].corr(df['yield_ton_per_hectare'])
corr_fert  = df['fertilizer_kg'].corr(df['yield_ton_per_hectare'])
print(f"\nQ8. Correlation with Yield:")
print(f"  rainfall_mm   : {corr_rain:.4f}")
print(f"  fertilizer_kg : {corr_fert:.4f}")
stronger = "rainfall_mm" if abs(corr_rain) > abs(corr_fert) else "fertilizer_kg"
print(f"  → '{stronger}' has a stronger relationship with yield.")

# Q9. Correlation Heatmap
print("\nQ9. Correlation Analysis:")
corr_matrix = df[num_cols].corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Q9. Correlation Heatmap")
plt.tight_layout()
plt.savefig("q9_heatmap.png")
plt.show()

yield_corr = corr_matrix['yield_ton_per_hectare'].drop('yield_ton_per_hectare').abs().sort_values(ascending=False)
print(f"\n  Top 3 features correlated with yield:")
print(yield_corr.head(3))

# Q10. Group-Based Analysis
print("\nQ10. Group-Based Analysis:")
crop_yield = df.groupby('crop_type')['yield_ton_per_hectare'].mean().sort_values(ascending=False)
soil_yield = df.groupby('soil_type')['yield_ton_per_hectare'].mean().sort_values(ascending=False)
print("\n  Average Yield by Crop Type:")
print(crop_yield)
print("\n  Average Yield by Soil Type:")
print(soil_yield)
print(f"\n  → Highest yield crop : {crop_yield.idxmax()} ({crop_yield.max():.2f} ton/ha)")
print(f"  → Highest yield soil : {soil_yield.idxmax()} ({soil_yield.max():.2f} ton/ha)")

# ─────────────────────────────────────────────
# PART C: Data Preparation
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART C: DATA PREPARATION")
print("=" * 60)

# Q11. Feature Encoding
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nQ11. Categorical Columns: {cat_cols}")

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
print("\n  First 5 rows after One-Hot Encoding:")
print(df_encoded.head())
print(f"\n  Shape after encoding: {df_encoded.shape}")

# Q12. Feature Selection
print("\nQ12. Feature Selection:")
X = df_encoded.drop(columns=['yield_ton_per_hectare'])
y = df_encoded['yield_ton_per_hectare']
print(f"  Target variable (y) : yield_ton_per_hectare")
print(f"  Input features  (X) : {list(X.columns)}")

# ─────────────────────────────────────────────
# PART D: Machine Learning
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART D: MACHINE LEARNING")
print("=" * 60)

# Q13. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nQ13. Train-Test Split (80/20):")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")

# Q14. Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

coef_df = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print(f"\nQ14. Linear Regression Model:")
print(f"  Intercept : {model.intercept_:.4f}")
print(f"\n  Coefficients:")
print(coef_df)
print(f"\n  → Feature with highest positive coefficient: '{coef_df.idxmax()}' ({coef_df.max():.4f})")
print(f"  R² Score (test set): {model.score(X_test, y_test):.4f}")
