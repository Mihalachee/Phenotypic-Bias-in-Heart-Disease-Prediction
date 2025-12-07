# %% [markdown]
# # **Phenotypic Bias in Heart Disease Prediction**
# ### *Fairness and Explainability Analysis using the UCI Heart Disease Dataset (Cleveland)*
# 
# **Author:** Francisca Mihalache  
# **Course:** Artificial Intelligence and Society — Individual Assignment 
# 
# ---
# 
# ## **1. Introduction and Context**
# 
# [cite_start]This notebook explores potential **phenotypic bias** in heart disease prediction models by analyzing whether performance differs across chest pain types (`cp`) — *typical angina*, *atypical angina*, *non-anginal pain*, and *asymptomatic* — in the **UCI Heart Disease (Cleveland)** dataset[cite: 1947].
# 
# The study integrates two key **Responsible AI** dimensions:
# * **Fairness:** Evaluating performance disparities between clinical subgroups.
# * **Explainability:** Using SHAP values to understand which features contribute to those differences.
# 
# [cite_start]This analysis adopts a **Data-Centric AI** perspective, emphasizing that model reliability depends primarily on the quality, representativeness, and interpretability of data rather than on algorithmic complexity[cite: 1947]. By addressing data intrinsic characteristics—such as phenotypic imbalance and demographic skew—this study aligns with the notion that responsible AI requires continuous data auditing and transparency.
# 
# ---
# 
# ## **Table of Contents**
# 
# 1. [Dataset Background and Structure](#1-dataset-background-and-structure)
#     * 1.1 [Dataset Loading and Description](#11-dataset-loading-and-description)
#     * 1.2 [Variable Definitions](#12-variable-definitions)
# 
# 2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
#     * 2.1 [General Information](#21-general-information)
#     * 2.2 [Missing Values](#22-missing-values)
#     * 2.3 [Chest Pain Type Distribution (cp)](#23-chest-pain-type-distribution-cp)
#     * 2.4 [Chest Pain Type vs. Sex](#24-chest-pain-type-vs-sex)
#         * 2.4.1 [Chest Pain Type vs. Sex -- Normalized between genders](#241-chest-pain-type-vs-sex----normalized-between-genders)
#     * 2.5 [Age Distribution by Sex](#25-age-distribution-by-sex)
#     * 2.6 [Correlation Matrix (Numeric Variables)](#26-correlation-matrix-numeric-variables)
#     * 2.7 [Pairplot](#27-pairplot)
#     * 2.8 [Missing Data Heatmap](#28-missing-data-heatmap)
#     * 2.9 [Feature Distributions by Disease Status](#29-feature-distributions-by-disease-status)
#     * 2.10 [Heart Disease Prevalence by Chest Pain Type](#210-heart-disease-prevalence-by-chest-pain-type)
#     * 2.11 [Age Distribution by Chest Pain Type](#211-age-distribution-by-chest-pain-type)
#     * 2.12 [Sex Distribution by Heart Disease Status](#212-sex-distribution-by-heart-disease-status)
#     * 2.13 [Age vs. Maximum Heart Rate (thalach)](#213-age-vs-maximum-heart-rate-thalach)
#     * 2.14 [Exercise Stress (oldpeak) by Chest Pain Type](#214-exercise-stress-oldpeak-by-chest-pain-type)
#     * 2.15 [Resting ECG Results by Heart Disease Status](#215-resting-ecg-results-by-heart-disease-status)
#     * 2.16 [Resting Blood Pressure vs. Maximum Heart Rate by Heart Disease Status](#216-resting-blood-pressure-vs-maximum-heart-rate-by-heart-disease-status)
# 
# 3. [Data Preprocessing](#3-data-preprocessing)
#     * 3.1 [Handling Missing Values](#31-handling-missing-values)
#     * 3.2 [Target Variable Transformation](#32-target-variable-transformation)
#     * 3.3 [Feature Encoding](#33-feature-encoding)
#     * 3.4 [Feature Scaling](#34-feature-scaling)
#     * 3.5 [Outlier Detection](#35-outlier-detection)
# 
# 4. [Model Training](#4-model-training)
#     * 4.1 [Data Splitting: Train / Validation / Test](#41-data-splitting-train--validation--test)
#     * 4.2 [Model Selection and Training](#42-model-selection-and-training)
#         * 4.2.1 [Logistic Regression](#421-logistic-regression)
#         * 4.2.2 [Random Forest Classifier](#422-random-forest-classifier)
#         * 4.2.3 [XGBoost Classifier](#423-xgboost-classifier)
#         * 4.2.4 [Multi-Layer Perceptron (MLP Classifier)](#424-multi-layer-perceptron-mlp-classifier)
#     * 4.3 [Performance and Fairness Analysis](#43-performance-and-fairness-analysis)
#         * 4.3.1 [Global Model Performance](#431-global-model-performance)
#             * 4.3.1.1 [Limitations of the Initial Train–Validation–Test Split](#4311-limitations-of-the-initial-trainvalidationtest-split)
#         * 4.3.2 [Gender-Specific Performance Analysis](#432-gender-specific-performance-analysis)
#             * 4.3.2.1 [Split the test set by gender](#4321-split-the-test-set-by-gender)
#             * 4.3.2.2 [Generate gender-specific predictions](#4322-generate-gender-specific-predictions)
#             * 4.3.2.3 [Compute standard performance metrics](#4323-compute-standard-performance-metrics)
#             * 4.3.2.4 [Compute fairness metrics](#4324-compute-fairness-metrics)
#             * 4.3.2.5 [Fairness Mitigation via Threshold Adjustment](#4325-fairness-mitigation-via-threshold-adjustment)
#         * 4.3.3 [Chest-Pain–Specific Performance Analysis](#433-chest-pain-specific-performance-analysis)
#         * 4.3.4 [Interaction: Chest Pain × Sex Performance Analysis](#434-interaction-chest-pain--sex-performance-analysis)
#     * 4.4 [SHAP-Based Explainability](#44-shap-based-explainability)
#         * 4.4.1 [Global SHAP](#441-global-shap)
#         * 4.4.2 [SHAP by Gender](#442-shap-by-gender)
#         * 4.4.3 [SHAP by Chest Pain Type](#443-shap-by-chest-pain-type)
#         * 4.4.4 [Local Explanations](#444-local-explanations)
# 
# 6. [Ethical Reflection and Discussion](#6-ethical-reflection-and-discussion)
#     * 6.1 [Phenotypic and gender bias in heart disease prediction](#61-phenotypic-and-gender-bias-in-heart-disease-prediction)
#     * 6.2 [Potential harms and societal impact](#62-potential-harms-and-societal-impact)
#     * 6.3 [Relation to the EU AI Act and regulatory classification](#63-relation-to-the-eu-ai-act-and-regulatory-classification)
#     * 6.4 [Responsible AI practices implemented in this study](#64-responsible-ai-practices-implemented-in-this-study)
#     * 6.5 [Remaining gaps and ethical recommendations](#65-remaining-gaps-and-ethical-recommendations)
# 
# 7. [References](#7-references)

# %% [markdown]
# #### Environment Setup
# 

# %%
pip install ucimlrepo

# %% [markdown]
# #### Imports

# %%
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
import missingno as msno

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# %% [markdown]
# <a id="1"></a>
# ## 1. Dataset Background and Structure
# 
# 

# %% [markdown]
# #### 1.1 Dataset Loading and Description
# We fetch the UCI Heart Disease (Cleveland) dataset using the ucimlrepo package.
# The dataset includes 303 patients and 13 clinical features related to cardiovascular risk.

# %%
# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Data as pandas DataFrames
X = heart_disease.data.features
y = heart_disease.data.targets

# Display concise metadata
meta_df = pd.DataFrame(list(heart_disease.metadata.items()), columns=["Key", "Value"])
display(meta_df)  

# Display only first 10 variable descriptions
display(heart_disease.variables)

# %% [markdown]
# The tables above summarize the **dataset metadata** and the **full list of variables** retrieved from the UCI Machine Learning Repository.  
# The original code example from the `ucimlrepo` documentation was slightly modified to present the information in a more compact and structured way.
# 
# From the **metadata table**, we confirm that:
# 
# - **Task type:** Classification (predicting presence of heart disease)  
# - **Instances:** 303 patients  
# - **Features:** 13 clinical attributes (a mix of categorical, integer, and real types)  
# - **Demographic variables:** Age and Sex  
# - **Target variable:** `num` — ranges from 0 (no disease) to 4 (presence of disease)  
# - **Missing values:** Yes — represented as `NaN`  
# - **Dataset creation year:** 1989, updated in 2023  
# - **Creators:** Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano  
# - **DOI:** [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)  
# - **Repository link:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
# 
# From the **variable table**, we can summarize:
# - There are 13 input features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`) and 1 target variable (`num`).  
# - Most features are either **integer** or **categorical**, aligning with clinical measurements and diagnostic categories.  
# - The `cp` (chest pain type) variable is especially important for this project — it differentiates patients by pain presentation (*typical angina*, *atypical angina*, *non-anginal pain*, *asymptomatic*) and will be used to study *phenotypic bias*.  
# - The variables `ca` (number of major vessels) and `thal` contain missing values.  
# - Descriptive units include `mm Hg` (for blood pressure) and `mg/dl` (for cholesterol).  
# 
# This metadata inspection confirms that the dataset is well-suited for the intended analysis on **fairness** and **explainability**.  
# 
# Next, we proceed to **data preprocessing**, where the target variable will be binarized and missing values will be handled before training the classification models.
# 

# %% [markdown]
# ### 1.2 Variable Definitions
# 
# According to the official documentation provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), the Heart Disease (Cleveland) dataset includes **14 clinical attributes** originally described by *Detrano et al. (1989)* in the *American Journal of Cardiology*.  
# 
# The variable definitions are summarized below and will be considered throughout this study for proper interpretation and clinical consistency.
# 
# | Feature | Description | Type / Units | Encoded Values |
# |----------|--------------|---------------|----------------|
# | **age** | Age of the patient | Integer / years | — |
# | **sex** | Biological sex | Categorical | 1 = male, 0 = female |
# | **cp** | Chest pain type | Categorical | 1 = typical angina<br>2 = atypical angina<br>3 = non-anginal pain<br>4 = asymptomatic |
# | **trestbps** | Resting blood pressure (on hospital admission) | Integer / mm Hg | — |
# | **chol** | Serum cholesterol level | Integer / mg/dl | — |
# | **fbs** | Fasting blood sugar > 120 mg/dl | Categorical | 1 = true, 0 = false |
# | **restecg** | Resting electrocardiographic results | Categorical | 0 = normal<br>1 = ST–T abnormality<br>2 = left ventricular hypertrophy |
# | **thalach** | Maximum heart rate achieved | Integer / bpm | — |
# | **exang** | Exercise-induced angina | Categorical | 1 = yes, 0 = no |
# | **oldpeak** | ST depression induced by exercise relative to rest | Float | — |
# | **slope** | Slope of the peak exercise ST segment | Categorical | 1 = upsloping<br>2 = flat<br>3 = downsloping |
# | **ca** | Number of major vessels colored by fluoroscopy | Integer (0–3) | — |
# | **thal** | Thalassemia | Categorical | 3 = normal<br>6 = fixed defect<br>7 = reversible defect |
# | **num** | Diagnosis of heart disease (angiographic disease status) | **Target variable** | 0 = no disease (< 50 % narrowing)<br>1–4 = disease present (> 50 % narrowing) |
# 
# These definitions form the foundation of all subsequent analyses.  
# They ensure that feature interpretations and clinical reasoning remain consistent with the original dataset documentation and medical context.
# 

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
# 
# After loading and describing the dataset, we now explore its main statistical properties and internal patterns to better understand how the features are distributed.

# %% [markdown]
# #### 2.1. General Information

# %%
X.info()

# Quick view of numeric variables
X.describe().T

# %% [markdown]
# The dataset consists of **303 patient records** and **13 measurable features** related to cardiovascular health.  
# All variables are numeric, meaning that categorical features (such as `sex`, `cp`, `restecg`, and `slope`) are encoded as integers.  
# This numeric representation simplifies model training but requires semantic remapping later for clearer interpretation.  
# 
# Two variables, `ca` (number of major vessels) and `thal` (thalassemia type), contain a small number of missing values.  
# These missing entries will be imputed before training to ensure data consistency.  
# 
# At this point, the dataset appears well-structured and suitable for the upcoming analyses.  
# The next steps will focus on exploring distributions, feature relationships, and possible data imbalances that may affect fairness across patient subgroups.
# 

# %% [markdown]
# #### 2.2. Missing Values

# %%
# Check for missing data
missing_summary = X.isna().sum()
missing_summary[missing_summary > 0]

# %% [markdown]
# As confirmed earlier in the variable table, only two features contain missing values: `ca` (number of major vessels) with **4 missing entries**, and `thal` (thalassemia type) with **2 missing entries**.  The number of missing values is very small compared to the dataset size, so these can be safely imputed later without affecting the statistical integrity of the data.
# 

# %%
# Convert the target variable to binary: 0 = No disease, 1 = Disease present
y_binary = y['num'].apply(lambda v: 1 if v > 0 else 0)

# Count the number of patients in each class
class_counts = y_binary.value_counts().sort_index()

# Print the class distribution
print("=== Heart Disease Case Distribution ===")
print(f"0 (No disease):  {class_counts[0]} patients")
print(f"1 (Has disease): {class_counts[1]} patients")
print(f"Total:            {class_counts.sum()} patients")

# Visualize the distribution
sns.countplot(x=y_binary, palette=['#1f77b4', '#ff7f0e'])
plt.title('Target Distribution: Presence of Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Number of Patients')
plt.show()


# %% [markdown]
# This near-balanced distribution (54% vs. 46%) indicates that class imbalance is minimal, allowing models to be trained and evaluated without requiring rebalancing techniques.  
# 
# In this context, techniques such as **SMOTE** or other oversampling methods are unnecessary and could even introduce artificial patterns not grounded in clinical reality.  
# Following the *Data-Centric AI* rationale, data balance should be evaluated in relation to its **semantic validity** rather than applied mechanically.  
# Since the Cleveland dataset already maintains a near-balanced distribution between healthy and diseased cases, preserving its natural proportions ensures more **faithful and interpretable** model behavior.
# 

# %% [markdown]
# #### 2.3. Chest Pain Type Distribution (cp)
# 
# Here we examine how the four **chest pain types (`cp`)** are distributed across the dataset.  
# This helps identify possible **phenotypic imbalance**, for example, whether some clinical presentations are much more common than others, which could later influence model fairness.

# %%
# Map chest pain codes to labels
cp_mapping = {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'}
X['cp_label'] = X['cp'].map(cp_mapping)

# Plot distribution of chest pain types
sns.countplot(x='cp_label', data=X, order=cp_mapping.values(), palette='Set2')
plt.title('Distribution of Chest Pain Types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Number of Patients')
plt.xticks(rotation=20)
plt.show()

# Print exact counts
print("Chest Pain Type Counts:")
print(X['cp_label'].value_counts())


# %% [markdown]
# The counts show that most patients are **asymptomatic (144)**, followed by **non-anginal pain (86)**,  
# while **typical angina (23)** is the least frequent.  
# This uneven distribution confirms the presence of **phenotypic imbalance**, meaning that models trained on this dataset may learn patterns dominated by asymptomatic and non-anginal cases.

# %% [markdown]
# The four chest pain categories used in the Cleveland dataset follow the clinical definitions adopted by Detrano et al. (1989):
# typical angina, atypical angina, non-anginal pain, and asymptomatic.
# This classification was a key clinical variable in the original study, as chest pain presentation was one of the most informative predictors of coronary artery disease.
# However, Detrano et al. noted that even asymptomatic and non-anginal patients could show significant angiographic abnormalities, revealing potential diagnostic bias when relying only on symptoms

# %% [markdown]
# #### 2.4. Chest Pain Type vs. Sex
# 
# This table compares **chest pain types (`cp`)** between **male** and **female** patients.  
# By observing these distributions, we can check whether certain pain types are more frequent in one sex — a relevant factor in understanding possible **gender-related diagnostic bias**.
# 

# %%
# Replace sex with readable labels
X['sex_label'] = X['sex'].map({0: 'Female', 1: 'Male'})

# Cross-tab and plot
sns.countplot(data=X, x='cp_label', hue='sex_label', order=cp_mapping.values(), palette='coolwarm')
plt.title('Chest Pain Types by Sex')
plt.xlabel('Chest Pain Type')
plt.ylabel('Number of Patients')
plt.legend(title='Sex')
plt.xticks(rotation=20)
plt.show()

# Print counts
pd.crosstab(X['cp_label'], X['sex_label'])


# %% [markdown]
# #### 2.4.1. Chest Pain Type vs. Sex -- Normalized between genders

# %%
# Create readable labels for sex variable
X['sex_label'] = X['sex'].map({0: 'Female', 1: 'Male'})

# Build a contingency table (absolute counts)
# Rows = sex, Columns = chest pain type (cp)
table = pd.crosstab(X['sex_label'], X['cp_label'])

# Convert absolute counts into proportions within each sex
# This removes the effect of having more males than females in the dataset
prop_df = table.div(table.sum(axis=1), axis=0)

# Convert table to long format for seaborn visualization
prop_df = prop_df.reset_index().melt(
    id_vars='sex_label',
    var_name='cp_label',
    value_name='proportion'
)

# Define custom colors to try to match previous plots-- I didn't find the exact colors used before
colors = {'Male': "#8198BF", 'Female': "#DB9872"}

# Final bar plot (proportion of each chest pain type within sex)
plt.figure(figsize=(8,6))
sns.barplot(
    data=prop_df,
    x='cp_label',
    y='proportion',
    hue='sex_label',
    hue_order=['Male', 'Female'],
    palette=colors
)

plt.title('Proportion of Chest Pain Types Within Each Sex')
plt.xlabel('Chest Pain Type')
plt.ylabel('Proportion (within each sex)')
plt.xticks(rotation=20)
plt.legend(title='Sex')
plt.show()


# %% [markdown]
# 
# The bar chart shows the **proportion of each chest pain type within each sex**, rather than absolute counts.  
# This normalization is important because the dataset contains **more male patients than female patients**, which would distort any comparison based solely on raw frequencies.
# 
# By plotting **proportions within each sex group**, we obtain a fair and unbiased comparison of symptom patterns:
# 
# - **Asymptomatic** chest pain remains the most common category for both men and women, but it is proportionally more frequent among men.
# - **Atypical angina** displays similar proportional levels in both groups.
# - **Typical angina** is relatively rare in both sexes.
# - **Non-anginal pain** shows the most important divergence.
# 
# **Women exhibit a higher proportion of *Non-anginal pain* than men once the data is normalized by sex.**
# 
# This difference was not visible when using raw counts because the male population is much larger.  
# The proportional analysis reveals a clinically meaningful pattern: female patients tend to present more non-anginal or atypical symptoms, which is consistent with medical literature on sex-specific cardiovascular symptomatology.
# 
# Normalizing the distribution therefore prevents misleading interpretations and ensures that comparisons between groups reflect **true clinical patterns rather than sampling imbalance**.
# 
# This pattern is consistent with evidence that women experience atypical or non-anginal symptoms more frequently than men in acute coronary syndromes, while men more commonly present with classic chest pressure patterns [7]
# 

# %% [markdown]
# #### 2.5. Age Distribution by Sex

# %%
sns.boxplot(x='sex_label', y='age', data=X, palette='pastel')
plt.title('Age Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Age (years)')
plt.show()

print(X.groupby('sex_label')['age'].describe()[['mean', 'std', 'min', 'max']])


# %% [markdown]
# The boxplot and summary statistics show that **female patients** have a slightly higher mean age (≈55.7 years) than **male patients** (≈53.8 years).  
# This reflects real-world clinical patterns, where women tend to develop heart disease later in life compared to men.

# %% [markdown]
# #### 2.6. Correlation Matrix (Numeric Variables)
# 
# The heatmap displays pairwise correlations between all numerical features.  

# %%
corr = X.corr(numeric_only=True)
plt.figure(figsize=(10,7))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix (Numeric Features)')
plt.show()


# %% [markdown]
# 
# Strong negative correlations are observed between **age** and **thalach** (−0.39), indicating that older patients generally reach lower maximum heart rates.  
# There is also moderate correlation between **oldpeak** and **slope** (≈0.58), consistent with how ST segment depression relates to exercise intensity.  
# No extreme multicollinearity is detected, suggesting that all variables can be safely used in modeling.

# %% [markdown]
# #### 2.7. Pairplot 
# The pairplot visualizes pairwise relationships between several numeric features, colored by **chest pain type (`cp`)**.  

# %%
sns.pairplot(X[['age','trestbps','chol','thalach','oldpeak','cp']], diag_kind='kde', hue='cp', palette='husl')
plt.suptitle('Pairwise Relationships by Chest Pain Type', y=1.02)
plt.show()


# %% [markdown]
# In general, *asymptomatic* patients (red points) tend to have **higher blood pressure and cholesterol**, which often corresponds to people who have underlying cardiovascular problems but do not feel obvious pain.  
# In contrast, those with *typical angina* (blue) usually have **clearer symptoms during effort** and show signs of better heart function, such as higher maximum heart rate and lower exercise-related stress (`oldpeak`).  
# 
# These patterns mirror what happens in real life: patients with fewer or no symptoms can still have more severe disease, which explains why models trained only on symptom intensity might misjudge these silent cases.  
# 
# This medical imbalance underlies the motivation for studying **phenotypic bias**.

# %% [markdown]
# #### 2.8. Missing Data Heatmap

# %%
msno.matrix(X)
plt.title('Missing Values Overview')
plt.show()

# %% [markdown]
# 
# The heatmap confirms that almost all variables are complete, except for a few missing entries in `ca` and `thal`.  
# This matches previous findings and confirms that missing data is minimal and localized, making simple imputation a suitable preprocessing strategy.

# %% [markdown]
# #### 2.9. Feature Distributions by Disease Status

# %% [markdown]
# 
# The boxplots compare key numeric variables between patients **with** and **without** heart disease.  

# %%
# Copy data and add binary target
df = X.copy()
df['Heart Disease'] = y_binary.map({0: 'No disease (0)', 1: 'Has disease (1)'})

# Select numeric features
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Reshape for plotting
df_melted = df.melt(id_vars='Heart Disease', value_vars=numeric_features,
                    var_name='Feature', value_name='Value')

# Boxplot
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Heart Disease', palette='Set1')
plt.title('Numeric Feature Distributions by Heart Disease Status')
plt.xlabel('Clinical Feature')
plt.ylabel('Measured Value')
plt.xticks(rotation=20)
plt.legend(title='Heart Disease Status')
plt.show()


# %% [markdown]
# 
# Those diagnosed with the condition generally have **lowerThis plot compares several health indicators between patients **with** and **without** heart disease.  
# People diagnosed with heart disease (blue boxes) tend to have:
# - Lower maximum heart rate (`thalach`), meaning their heart cannot reach high activity levels;
# - Higher `oldpeak` values, which reflect greater stress during exercise tests;
# - Slightly higher cholesterol (`chol`) and resting blood pressure (`trestbps`).
# 
# These differences are consistent with how heart disease usually appears in practice patients with cardiovascular problems often have limited heart performance and higher blood pressure or cholesterol.  
# 
# This confirms that the dataset captures realistic physiological differences between healthy and affected individuals.
# 

# %% [markdown]
# #### 2.10. Heart Disease Prevalence by Chest Pain Type
# 
# This plot examines the **proportion of patients with heart disease** within each chest pain category.  
# It helps identify whether certain pain types — such as *asymptomatic* or *non-anginal* — are associated with higher disease prevalence, which is central to understanding **phenotypic bias**.
# 

# %%
df['Heart Disease Num'] = df['Heart Disease'].map({'Has disease (1)': 1, 'No disease (0)': 0})


sns.barplot(
    data=df,
    x='cp_label',
    y='Heart Disease Num',
    order=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
    palette='Set2',
    estimator='mean'
)
plt.title('Heart Disease Prevalence by Chest Pain Type')
plt.ylabel('Proportion with Disease')
plt.xlabel('Chest Pain Type')
plt.ylim(0, 1)
plt.show()


# %% [markdown]
# As seen, **asymptomatic patients** have the highest prevalence (around 75%), while **typical angina** and **atypical angina** show much lower rates (below 30%).  
# This means that people who **do not experience clear pain** are often the ones **with the most severe disease**. A silent but dangerous pattern also described in the clinical literature.  
# 
# It highlights how relying only on pain as a symptom can lead to underdiagnosis of serious cases.

# %% [markdown]
# #### 2.11. Age Distribution by Chest Pain Type
# 
# This plot compares **age** across the four chest pain types.  
# It helps determine whether older patients are more likely to present atypical or asymptomatic pain,  
# which can contribute to bias when models rely on symptoms rather than physiological measures.

# %%
sns.boxplot(
    data=df,
    x='cp_label',
    y='age',
    order=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
    palette='pastel'
)
plt.title('Age Distribution by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Age (years)')
plt.show()


# %% [markdown]
# 
# The boxplots indicate that **asymptomatic** and **typical angina** patients are generally older, while **atypical** and **non-anginal pain** groups include younger individuals.  
# 
# This aligns with the fact that **age increases the risk** of coronary artery disease, and that older patients often develop **fewer or atypical symptoms**, which complicates early detection.

# %% [markdown]
# #### 2.12. Sex Distribution by Heart Disease Status
# 
# This plot visualizes how **male** and **female** patients are distributed across heart disease outcomes.  
# It highlights potential **gender imbalances**, which may affect both medical interpretation and model fairness.
# 

# %%
sns.countplot(
    data=df,
    x='Heart Disease',
    hue='sex_label',
    palette='coolwarm'
)
plt.title('Sex Distribution by Heart Disease Status')
plt.xlabel('Heart Disease Status')
plt.ylabel('Number of Patients')
plt.legend(title='Sex')
plt.show()


# %% [markdown]
# This bar chart shows that **men** represent the majority of patients, especially among those diagnosed with heart disease.
# 
# Women appear less frequently in the dataset and tend to have **atypical or asymptomatic pain**, a pattern consistent with medical findings: heart disease in women is often **under-recognized** because symptoms differ from those in men.

# %% [markdown]
# #### 2.13. Age vs. Maximum Heart Rate (thalach)
# 
# This scatter plot explores the relationship between **age** and **maximum heart rate achieved**,  
# colored by heart disease status.  
# It provides insight into how cardiovascular performance decreases with age,  
# and whether this decline differs between healthy and affected individuals.
# 

# %%
sns.scatterplot(
    data=df,
    x='age',
    y='thalach',
    hue='Heart Disease',
    palette='Set1'
)
plt.title('Age vs. Maximum Heart Rate by Heart Disease Status')
plt.xlabel('Age (years)')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.show()


# %% [markdown]
# The scatter plot shows a clear negative relationship between **age** and **maximum heart rate** : older patients naturally reach lower heart rates during exercise.  
# 
# Those **with heart disease** (blue points) usually achieve **lower heart rates** than healthy individuals, indicating reduced cardiovascular efficiency and confirming expected physiological decline with disease severity.

# %% [markdown]
# #### 2.14. Exercise Stress (oldpeak) by Chest Pain Type
# 
# This boxplot compares the **ST depression (`oldpeak`)** among different chest pain types.  
# Higher `oldpeak` values usually indicate greater exercise-induced stress on the heart,  
# making it an important clinical indicator for disease severity.
# 

# %%
sns.boxplot(
    data=df,
    x='cp_label',
    y='oldpeak',
    order=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
    palette='Set3'
)
plt.title('Exercise Stress (oldpeak) by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('ST Depression (oldpeak)')
plt.show()


# %% [markdown]
# 
# This boxplot compares exercise-induced stress (ST depression, measured as `oldpeak`) across chest pain categories.  
# 
# **Asymptomatic** patients show the **highest oldpeak values**, meaning their hearts experience stronger stress during exercise despite the absence of pain.  
# 
# This paradox: severe physiological stress with no pain reinforces the concern that **silent or atypical cases** may represent advanced but undetected heart disease.

# %% [markdown]
# #### 2.15. Resting ECG Results by Heart Disease Status
# 
# This plot examines how **resting electrocardiogram (ECG) results** (`restecg`) relate to heart disease presence.  
# 
# In the Cleveland study (Detrano et al., 1989), this variable was one of the original predictors used to estimate the probability of coronary artery disease.  
# 
# Although not highly discriminative on its own, ECG patterns, particularly **ST–T abnormalities** and **left ventricular hypertrophy**, were found to add diagnostic value when combined with exercise-related features.
# 

# %%
sns.countplot(data=df, x='restecg', hue='Heart Disease', palette='Set2')
plt.title('Resting ECG Results by Heart Disease Status')
plt.xlabel('Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)')
plt.ylabel('Number of Patients')
plt.legend(title='Heart Disease')
plt.show()


# %% [markdown]
# The **resting electrocardiogram (ECG)** measures the heart’s electrical activity at rest.  
# In this dataset, the variable `restecg` has three categories:
# 
# - **0 = Normal ECG:** No significant electrical abnormalities.  
# - **1 = ST–T Abnormality:** Distorted ST or T waves, often signaling ischemia or lack of oxygen supply.  
# - **2 = Left Ventricular Hypertrophy (LVH):** Enlargement of the left ventricle, typically due to prolonged high blood pressure.
# 
# The plot shows that patients with **LV hypertrophy (code 2)** or **ST–T abnormalities (code 1)** are more likely to have heart disease than those with normal readings.  
# 
# While most patients in the dataset fall into the “normal” or “LVH” categories, the **disease ratio rises sharply** for the latter.
# 
# This matches the findings of **Detrano et al. (1989)**, who noted that although ECG data alone were not the most powerful predictor, they provided *baseline evidence of structural or electrical stress on the heart* that complemented other noninvasive indicators (exercise ECG, thallium scans, and fluoroscopy).

# %% [markdown]
# #### 2.16. Resting Blood Pressure vs. Maximum Heart Rate by Heart Disease Status
# 
# This scatter plot explores the relationship between **resting blood pressure (`trestbps`)** and **maximum heart rate achieved (`thalach`)** during exercise.  
# Detrano et al. (1989) found that these two measurements, when analyzed jointly with age,  
# were significant in distinguishing between healthy and diseased patients — reflecting both cardiovascular efficiency and response to stress.
# 

# %%
sns.scatterplot(data=df, x='trestbps', y='thalach', hue='Heart Disease', palette='coolwarm')
plt.title('Resting Blood Pressure vs. Maximum Heart Rate by Heart Disease Status')
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.show()


# %% [markdown]
# This scatter plot compares **resting systolic blood pressure (`trestbps`)**, measured in mmHg at hospital admission, 
# with the **maximum heart rate achieved (`thalach`)** during the exercise test.  
# 
# In a healthy cardiovascular system:
# - Blood pressure at rest tends to stay moderate (around 120 mmHg).  
# - The heart rate rises efficiently during exercise (high `thalach` values).  
# 
# The pattern seen here reflects the opposite trend for patients with heart disease: those with **higher resting blood pressure** tend to achieve **lower maximum heart rates**.  
# 
# They cluster in the upper-left region of the chart (high pressure, low performance),  
# indicating **limited cardiac efficiency** and possible arterial stiffness.
# 
# This physiological relationship was central to the Cleveland study, **Detrano et al. (1989)** reported that age, resting blood pressure, and exercise capacity together formed a powerful combination for predicting angiographically confirmed coronary artery disease (CAD).  
# 
# These results visually confirm that the same hemodynamic imbalance persists in this dataset.

# %% [markdown]
# ## 3. Data Preprocessing
# 
# Before modeling, it is essential to clean and standardize the dataset to ensure that no technical artifacts influence the fairness or explainability analysis. 
# 
# Following the original methodology described by **Detrano et al. (1989)**, data preprocessing involves imputing missing values, transforming categorical variables, and normalizing continuous features while preserving their clinical meaning.  
# This step ensures the input data remain consistent with the structure of the Cleveland diagnostic model.
# 
# These preprocessing steps are not just technical adjustments but deliberate **data-centric interventions** aimed at improving dataset reliability.  
# Following the *DC-AI principle of continuous data improvement*, each transformation — from imputing missing values to scaling continuous features — is guided by domain knowledge rather than automated heuristics.  
# This aligns with the notion that high-quality, interpretable data form the foundation of trustworthy AI systems.
# 

# %% [markdown]
# #### 3.1. Handling Missing Values
# 
# In the original Cleveland study (Detrano et al., 1989), these variables correspond to the three non-invasive diagnostic tests performed on all patients:
# - `oldpeak` — exercise electrocardiogram (ST depression)
# - `thal` — thallium scintigraphy
# - `ca` — cardiac fluoroscopy (number of calcified vessels)
# 
# The results of these tests were **not used to guide clinical decisions**, as angiography remained the gold standard.  
# Their inclusion here allows the model to simulate the same diagnostic reasoning process analyzed by Detrano et al. (1989).  
# 
# Imputing missing values in `ca` and `thal` ensures these critical imaging indicators are preserved for the fairness and explainability analysis.
# 
# Because missingness is limited and likely not systematic (no demographic pattern observed), we replace missing entries with their **mode values** to preserve distributional integrity.
# 

# %%
df.fillna({'ca': df['ca'].mode()[0], 'thal': df['thal'].mode()[0]}, inplace=True)

print("Missing values after imputation:")
print(df[['ca', 'thal']].isna().sum())

# %% [markdown]
# #### 3.2. Target Variable Transformation
# 
# In the original Cleveland dataset, the variable `num` ranges from 0 to 4, indicating the **severity of coronary artery narrowing** (_0 = no disease_, _4 = severe disease_), as defined by Detrano et al. (1989).  
# 
# For modeling purposes, we create a new binary target variable that captures **disease presence** rather than severity.  
# Values 1–4 are grouped as _“disease present”_ (`1`), while 0 represents _“no disease”_ (`0`).  
# 
# This binary target (`target`) is used for classification and fairness analysis, while the original `num` values are preserved as `severity` for interpretability and post-hoc analysis.
# 

# %%
# Keep the original severity levels (0–4)
df['severity'] = y['num'].astype(int)

# Create binary target for classification
df['target'] = df['severity'].apply(lambda x: 1 if x > 0 else 0)

print("Unique values in severity:", df['severity'].unique())
print("Binary target distribution:")
print(df['target'].value_counts())

# %% [markdown]
# #### 3.3. Feature Encoding
# 
# In the Cleveland dataset, several clinical attributes are **categorical**, representing discrete diagnostic categories rather than continuous measurements.  
# To make them compatible with machine learning algorithms, we convert these variables into numerical representations.  
# 
# The key categorical features are:  
# - `sex` — biological sex (0 = female, 1 = male)  
# - `cp` — chest pain type (1–4)  
# - `restecg` — resting electrocardiographic results (0–2)  
# - `slope` — slope of the ST segment during exercise (1–3)  
# - `thal` — thallium stress test result (3 = normal, 6 = fixed defect, 7 = reversible defect)
# 
# Following Detrano et al. (1989), these features encode **qualitative physiological states** that strongly influence disease probability.  
# To preserve their interpretability while making them model-ready, we apply **one-hot encoding**.  
# This avoids imposing an artificial order (e.g., “3 > 1”) among non-ordinal categories.
# 
# 

# %%
# Select columns that are categorical or boolean
cat_cols = df.select_dtypes(include=['object', 'bool']).columns

print("Detected categorical columns:\n", list(cat_cols))

# Show unique values for each categorical column
for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

# %%
df['Heart Disease'] = df['Heart Disease'].map({
    'No disease (0)': 0,
    'Has disease (1)': 1
})

# Encode sex
df['sex_label'] = df['sex_label'].map({
    'Female': 0,
    'Male': 1
})

# One-hot encode chest pain labels (keep all categories)
df = pd.get_dummies(df, columns=['cp_label'], drop_first=False)

# Confirm everything is now numeric
print(df.dtypes)

# %% [markdown]
# #### 3.4. Feature Scaling
# 
# The numerical variables have **different units and magnitudes**, for instance, `age` is measured in years, `chol` in mg/dL, and `oldpeak` in millivolts.  
# This difference can distort distance-based algorithms and affect the convergence of gradient-based models.  
# 
# To ensure all features contribute equally, we apply **standardization** using `StandardScaler`, which rescales continuous variables to have **zero mean and unit variance**.  
# This approach maintains comparability while preserving relationships between variables.
# 

# %%
from sklearn.preprocessing import StandardScaler

# Select continuous variables for scaling
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Initialize and apply scaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Scaled numerical columns:")
print(numeric_cols)

# %% [markdown]
# #### 3.5. Outlier Detection
# 
# Outliers in clinical data can arise from **measurement errors** or **rare pathological conditions**.  
# Detrano et al. (1989) used carefully curated hospital records, so extreme outliers are uncommon, but identifying them helps ensure data integrity.  
# 
# We use the **Interquartile Range (IQR)** method to flag potential outliers without removing clinically valid cases.  
# The goal is to detect unusual deviations in features such as cholesterol (`chol`), resting blood pressure (`trestbps`), and ST depression (`oldpeak`).
# 

# %%
# Identify outliers using IQR method
for col in ['trestbps', 'chol', 'oldpeak']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outliers} potential outliers detected.")

# %% [markdown]
# A small number of potential outliers were detected in `trestbps`, `chol`, and `oldpeak`.  
# These deviations likely represent real physiological variability rather than measurement errors, since the Cleveland dataset originated from controlled hospital environments.  
# Consistent with the 1989 study, these values are retained to preserve the clinical diversity of the sample.
# 

# %% [markdown]
# ## 4. Model Training
# 
# After data preprocessing, the dataset is now clean, standardized, and ready for model training.  
# This section focuses on building and comparing several **black-box models** to predict the presence of heart disease.  
# The goal is to evaluate their overall performance and later analyze **phenotypic fairness** and **explainability**.
# 
# The training procedure follows the **Data-Centric AI** mindset:
# - ensure consistent data splits for unbiased evaluation,  
# - compare multiple models differing in complexity,  
# - select the most reliable one for fairness and SHAP interpretation.
# 

# %% [markdown]
# ### 4.1 Data Splitting: Train / Validation / Test
# 
# To ensure a fair evaluation, the dataset is divided into three subsets:
# - **Training (70 %)** – used to fit the models;  
# - **Validation (15 %)** – used to tune and compare models;  
# - **Test (15 %)** – held out for the final, unbiased assessment.
# 
# Stratification is applied to maintain the same class proportion across all subsets.

# %%
from sklearn.model_selection import train_test_split

# Define target and features
leak_cols = ['Heart Disease', 'Heart Disease Num', 'severity', "target", "num"]
X = df.drop(columns=[c for c in leak_cols if c in df.columns])
y = df['target']

# --- Split once and reuse for all models ---
# 70% train, 30% temporary (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Split remaining 30% equally into validation and test (15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Confirm class proportions remain balanced
print("\nClass distribution:")
for name, target in zip(["Train", "Validation", "Test"], [y_train, y_val, y_test]):
    print(f"{name} - Disease: {target.mean():.2%}")


# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
leak_cols = ['Heart Disease', 'Heart Disease Num', 'severity', 'target', "num"]
X = df.drop(columns=[c for c in leak_cols if c in df.columns])
y = df['Heart Disease']

# Split FIRST
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# THEN preprocess inside a pipeline
num_cols = ['age','trestbps','chol','thalach','oldpeak','ca']
cat_cols = ['sex','cp','fbs','restecg','exang','slope','thal']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# %% [markdown]
# ### 4.2. Model Selection and Training
# 
# To evaluate how different learning paradigms respond to the complexity and fairness challenges of clinical data, four machine learning models were selected: **Logistic Regression**, **Random Forest**, **XGBoost** and a **Multi-Layer Perceptron**. This selection reflects current findings in the literature, where ensemble-based models like Random Forest and XGBoost have shown superior accuracy on the Cleveland dataset (up to 97–98% AUC), while simpler models such as Logistic Regression remain competitive and highly interpretable.
# 
# Each model represents a distinct learning bias: linear (LR), ensemble bagging (RF), boosting (XGB), and deep neural representations (MLP). This diversity enables systematic comparison of:
# - **Predictive performance under data overlap and imbalance**, 
# - **Explainability using SHAP**, and 
# - **Fairness across subgroups**, as explored in M03 and M04.
# 
# From a **data-centric perspective**, the goal is not to find a single “best” algorithm, but to observe how model behavior varies given the quality and structure of the data. This follows the principle that robust and fair predictions emerge from **data quality improvements** and model accountability—not algorithm complexity alone.

# %% [markdown]
# #### 4.2.1. Logistic Regression
# 
# **Logistic Regression (LR)** was used as a baseline model due to its simplicity, interpretability, and widespread application in clinical risk prediction. It assumes a linear relationship between the input features and the log-odds of the binary outcome, making it especially useful for identifying linearly separable patterns in the data [1].
# 
# In this experiment, L2 regularization was applied to reduce overfitting, and `class_weight='balanced'` was used to mitigate the mild class imbalance. The model was trained using the same training/validation/test split to ensure fair comparison with other classifiers. Evaluation metrics included accuracy, F1 score (macro and weighted), recall, precision, and log loss.
# 
# Although its capacity to capture complex interactions is limited, LR provides a transparent benchmark for fairness and explainability analysis, especially when compared to more opaque models such as tree ensembles or neural networks [2].

# %%
# -------------------------
# 1. Define evaluation function
# -------------------------
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision (macro)": precision_score(y, y_pred, average='macro'),
        "Recall (macro)": recall_score(y, y_pred, average='macro'),
        "F1 (macro)": f1_score(y, y_pred, average='macro'),
        "F1 (weighted)": f1_score(y, y_pred, average='weighted'),
        "Log Loss": log_loss(y, y_proba),
    }
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Has Disease'])
    fig, ax = plt.subplots(figsize=(3, 3))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix — {dataset_name}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return metrics

# -------------------------
# 2. Train Logistic Regression
# -------------------------
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    penalty='l2',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train, y_train)

# -------------------------
# 3. Evaluate on all sets
# -------------------------
results = []
results.append(evaluate_model(lr_model, X_train, y_train, "Logistic Regression (Train Set)"))
results.append(evaluate_model(lr_model, X_val, y_val, "Logistic Regression (Validation Set)"))
results.append(evaluate_model(lr_model, X_test, y_test, "Logistic Regression (Test Set)"))

# -------------------------
# 4. Combine metrics
# -------------------------
results_lr = pd.DataFrame(results).set_index("Dataset")
display(results_lr.round(4))


# %% [markdown]
# #### 4.2.2. Random Forest Classifier
# 
# The **Random Forest** model serves as a strong baseline.  
# 
# It combines multiple decision trees to reduce variance and improve generalization.  
# It is also compatible with SHAP, which will later be used for explainability analysis.
# 

# %%
# -------------------------
# 1. Train Random Forest
# -------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
)
import pandas as pd
import matplotlib.pyplot as plt

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -------------------------
# 2. Evaluation function
# -------------------------
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision (macro)": precision_score(y, y_pred, average='macro'),
        "Recall (macro)": recall_score(y, y_pred, average='macro'),
        "F1 (macro)": f1_score(y, y_pred, average='macro'),
        "F1 (weighted)": f1_score(y, y_pred, average='weighted'),
        "Log Loss": log_loss(y, y_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Has Disease'])
    fig, ax = plt.subplots(figsize=(3, 3))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix — Random Forest ({dataset_name})", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return metrics

# -------------------------
# 3. Evaluate on all sets
# -------------------------
results = []
results.append(evaluate_model(rf_model, X_train, y_train, "Random Forest (Train Set)"))
results.append(evaluate_model(rf_model, X_val, y_val, "Random Forest (Validation Set)"))
results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest (Test Set)"))

# -------------------------
# 4. Combine metrics into table
# -------------------------
results_rf = pd.DataFrame(results).set_index("Dataset")
display(results_rf.round(4))


# %% [markdown]
# The overall performance is clinically meaningful and statistically stable, making this model a reliable baseline for heart disease risk prediction.

# %% [markdown]
# #### 4.2.3. XGBoost Classifier
# 
# The **Extreme Gradient Boosting (XGBoost)** algorithm extends the concept of ensemble trees by using a **boosting strategy** instead of bagging.  
# While Random Forest builds many independent trees and averages their predictions, XGBoost builds trees *sequentially* — each new tree focuses on correcting the residual errors of the previous ones.  
# This iterative process allows XGBoost to capture complex nonlinear interactions among clinical features, often achieving higher predictive accuracy.
# 
# From a **Data-Centric AI** perspective, XGBoost is valuable because it can:
# - Handle **imbalanced data** using scale adjustments (e.g., `scale_pos_weight`);
# - Automatically manage **missing values** by learning optimal splits;
# - Provide **feature importance** scores, enabling interpretability analysis;
# - Adapt to **heterogeneous data quality**, common in medical datasets.
# 
# The model was trained using the same training, validation, and test splits to ensure fair comparison with the previous classifiers.
# 

# %%
# -------------------------
# 1. Train XGBoost Classifier
# -------------------------
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
)
import pandas as pd
import matplotlib.pyplot as plt
# -------------------------
# 1. Define evaluation function (reusable for all models)
# -------------------------
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    
    
    try:
        y_proba = model.predict_proba(X)
    except AttributeError:
        y_proba = [[1 - p, p] for p in y_pred]  # fallback (not ideal but avoids error)
    
    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision (macro)": precision_score(y, y_pred, average='macro'),
        "Recall (macro)": recall_score(y, y_pred, average='macro'),
        "F1 (macro)": f1_score(y, y_pred, average='macro'),
        "F1 (weighted)": f1_score(y, y_pred, average='weighted'),
        "Log Loss": log_loss(y, y_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Has Disease'])
    fig, ax = plt.subplots(figsize=(3, 3))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix — {dataset_name}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return metrics

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# -------------------------
# 2. Evaluate on all sets
# -------------------------
results = []
results.append(evaluate_model(xgb_model, X_train, y_train, "XGBoost (Train Set)"))
results.append(evaluate_model(xgb_model, X_val, y_val, "XGBoost (Validation Set)"))
results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost (Test Set)"))

# -------------------------
# 3. Combine metrics
# -------------------------
results_xgb = pd.DataFrame(results).set_index("Dataset")
display(results_xgb.round(4))


# %% [markdown]
# #### 4.2.4. Multi-Layer Perceptron (MLP Classifier)
# 
# The **Multi-Layer Perceptron (MLP)** is a type of feed-forward artificial neural network designed to model complex nonlinear relationships between input features.  
# Unlike linear models such as Logistic Regression or tree-based ensembles like Random Forest and XGBoost, the MLP learns hidden feature interactions through **layers of interconnected neurons**, each applying nonlinear activation functions.
# 
# In this study, the MLP is used to assess how deep learning architectures perform on the same dataset and whether their higher representational power improves predictive accuracy.  
# From a **Data-Centric AI** perspective, neural networks are valuable for detecting subtle feature patterns but may also amplify existing data biases if not properly balanced.  
# Thus, the comparison between the MLP and traditional models provides insight into the trade-off between accuracy, interpretability, and fairness.
# 

# %%
# -------------------------
# 1. Train MLP Classifier
# -------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
)
import pandas as pd
import matplotlib.pyplot as plt

mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,                   # L2 regularization
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
mlp_model.fit(X_train, y_train)

# -------------------------
# 2. Evaluate on all sets
# -------------------------
results = []
results.append(evaluate_model(mlp_model, X_train, y_train, "MLP (Train Set)"))
results.append(evaluate_model(mlp_model, X_val, y_val, "MLP (Validation Set)"))
results.append(evaluate_model(mlp_model, X_test, y_test, "MLP (Test Set)"))

# -------------------------
# 3. Combine metrics
# -------------------------
results_mlp = pd.DataFrame(results).set_index("Dataset")
display(results_mlp.round(4))


# %% [markdown]
# ### 4.3. Performance and Fairness Analysis       
# This section evaluates how each trained model performs across different demographic groups (e.g., male vs female).  
# The goal is to identify whether certain subpopulations experience systematic disparities in model predictions, reflecting issues of **algorithmic bias**.
# 
# Performance is analyzed along two dimensions:
# 1. **Overall effectiveness** — how each model generalizes (Accuracy, F1, Log Loss);
# 2. **Equity across groups** — whether prediction quality differs significantly by gender.
# 
# Following the **Data-Centric AI** principles introduced earlier, this analysis connects model behavior to data quality, representation imbalance, and ethical implications in medical AI.

# %% [markdown]
# #### 4.3.1. Global Model Performance
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Combine all model results into a single table
summary_all = pd.concat([
    results_lr,
    results_rf,
    results_xgb,
    results_mlp
])

# Show all sets (train, val, test)
display(summary_all.round(4))

# Filter only the test results for fair comparison
summary_test = summary_all[summary_all.index.str.contains("Test")]
display(summary_test.round(4))
summary_test[['Accuracy', 'F1 (macro)']].plot(
    kind='bar',
    figsize=(12, 4),
    title='Model Comparison on Test Set',
    ylabel='Score',
    colormap='viridis'
)

plt.xticks(rotation=45)
plt.ylim(0, 1.05) 

for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f', label_type='edge', fontsize=9)

plt.tight_layout()
plt.show()
plt.show()

# %% [markdown]
# More complex models (RF, XGBoost, MLP) did not outperform the well-structured Logistic Regression, confirming that improving data representation and balance often has greater impact than increasing model complexity

# %% [markdown]
# In the previous configuration, the XGBoost model showed near-perfect performance on the training set (Accuracy = 1.00) but significantly lower results on validation and test data.
# Such a discrepancy strongly indicated overfitting, meaning the model memorized training patterns rather than learning generalizable relationships.
# This was particularly unexpected since simpler models like Logistic Regression achieved more stable and higher generalization scores.
# To address this issue, the model was retrained using stronger regularization and early stopping mechanisms.
# The following configuration reduces tree depth, applies both L1 and L2 penalties, and monitors validation loss during training.
# The process stops automatically when no further improvement is observed for 50 consecutive rounds, preventing unnecessary boosting iterations.

# %%
import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.03,
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'seed': 42
}

# Train with early stopping
xgb_tuned_native = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=False
)

# Make predictions
y_pred = (xgb_tuned_native.predict(dtest) >= 0.5).astype(int)

class XGBWrapper:
    def __init__(self, booster):
        self.model = booster
    def predict_proba(self, X):
        return np.vstack([1 - self.model.predict(xgb.DMatrix(X)), self.model.predict(xgb.DMatrix(X))]).T
    def predict(self, X):
        return (self.model.predict(xgb.DMatrix(X)) >= 0.5).astype(int)

xgb_tuned = XGBWrapper(xgb_tuned_native)
res_tuned = []
res_tuned.append(evaluate_model(xgb_tuned, X_train, y_train, "XGB Tuned (Train)"))
res_tuned.append(evaluate_model(xgb_tuned, X_val, y_val, "XGB Tuned (Val)"))
res_tuned.append(evaluate_model(xgb_tuned, X_test, y_test, "XGB Tuned (Test)"))

df_xgb_tuned = pd.DataFrame(res_tuned).set_index("Dataset")
display(df_xgb_tuned.round(4))
print("Best iteration:", xgb_tuned_native.best_iteration)


# %% [markdown]
# After applying stronger regularization and early stopping, the tuned XGBoost model achieved a much healthier balance between learning capacity and generalization.
# Training accuracy dropped from 1.00 to 0.896, indicating that overfitting was successfully mitigated.
# 
# More importantly, the test accuracy increased to 0.891, outperforming the initial version while maintaining a lower log loss (0.308), showing improved probability calibration.
# The validation score (0.778) remains slightly lower, consistent with early stopping behavior the model halts before overfitting to the validation set.
# 
# Overall, this configuration demonstrates that proper regularization can outperform excessive model complexity, confirming that performance improvements come from controlling variance rather than increasing depth or boosting rounds.

# %% [markdown]
# These global metrics assess aggregate model performance but do not reveal whether predictive quality is consistent across demographic subgroups. Following the Data-Centric AI workflow and the fairness framework introduced earlier, we now examine whether the model behaves similarly for male and female patients, given the known sex-specific differences in chest pain presentation.

# %% [markdown]
# ##### 4.3.1.1. Limitations of the Initial Train-Validation-Test-Split
# 
# The initial experimental setup relied on a conventional 70–15–15 train–validation–test division. This strategy is commonly used in larger datasets, where each subset contains a sufficient number of samples to provide representative and stable estimates of model performance. However, the Cleveland dataset is relatively small, comprising fewer than 300 instances after preprocessing. As a consequence, the validation and test partitions contained only a few dozen samples each. With such limited data, performance estimates become highly sensitive to sampling variability, and small fluctuations in the composition of each subset can lead to disproportionately large differences in accuracy, F1-score, or error rates.
# 
# This limitation became evident during the subgroup analyses. When performance was evaluated separately for male and female patients or across different chest pain categories, the extremely small number of samples within each subgroup generated unstable metrics and hindered the ability to draw reliable conclusions. The discrepancy observed earlier between simple models, such as Logistic Regression, and more complex models, such as XGBoost, further illustrated the impact of relying on a single arbitrary split. In particular, the initial configuration misleadingly suggested that XGBoost generalised more poorly than expected, an effect attributable not to the model itself but to the small size and sensitivity of the validation and test sets.
# 
# Given these observations, and following methodological principles emphasised in the course material concerning data scarcity, sampling noise, and robust evaluation, it became necessary to *revise the validation strategy*.
# 
# To obtain a more reliable estimate of true generalisation performance, the final XGBoost model was re-evaluated using **stratified K-fold cross-validation**. This approach distributes the available samples more effectively, ensures that each instance is used both for training and testing, and reduces variance in performance estimates. The transition to cross-validation therefore reflects the practical constraints imposed by the dataset and provides a more rigorous foundation for the subsequent fairness and subgroup analyses.
# 

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_cols = ['age','trestbps','chol','thalach','oldpeak','ca']
cat_cols = ['sex','cp','fbs','restecg','exang','slope','thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)


# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import numpy as np
import xgboost as xgb
import pandas as pd

# Convert target to binary (0 = no disease, 1 = disease)
y = (df["Heart Disease"] > 0).astype(int)

# Definition of fold evaluation
def evaluate_fold(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro"),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted"),
        "Log Loss": log_loss(y_true, y_proba)
    }

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store all fold indices
fold_indices = list(skf.split(X, y))

xgb_metrics = []
best_iterations = []

# Loop through each fold as the TEST fold
for i in range(5):
    print(f"Running fold {i+1}/5 ...")

    # ------------------------
    # 1. Select TEST fold
    # ------------------------
    test_idx = fold_indices[i][1]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    # ------------------------
    # 2. Select VALIDATION fold
    #    We choose the next fold (cyclic)
    # ------------------------
    val_fold = (i + 1) % 5
    val_idx = fold_indices[val_fold][1]

    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # ------------------------
    # 3. Remaining 3 folds → TRAIN
    # ------------------------
    train_folds = [j for j in range(5) if j not in [i, val_fold]]
    train_idx = np.concatenate([fold_indices[j][1] for j in train_folds])

    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    # ------------------------
    # 4. Preprocessing
    # ------------------------
    X_train_proc = preprocessor.fit_transform(X_train_fold)
    X_val_proc   = preprocessor.transform(X_val_fold)
    X_test_proc  = preprocessor.transform(X_test_fold)

    dtrain = xgb.DMatrix(X_train_proc, label=y_train_fold)
    dval   = xgb.DMatrix(X_val_proc,   label=y_val_fold)
    dtest  = xgb.DMatrix(X_test_proc,  label=y_test_fold)

    # ------------------------
    # 5. Model training with early stopping
    # ------------------------
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_iterations.append(booster.best_iteration)

    # ------------------------
    # 6. Evaluation on TEST fold
    # ------------------------
    y_proba = booster.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)

    xgb_metrics.append(evaluate_fold(y_test_fold, y_pred, y_proba))

# ------------------------
# 7. Final Summary
# ------------------------
df_results = pd.DataFrame(xgb_metrics)
print("Cross-validated metrics (3–1–1 Scheme):")
display(df_results.round(4))

print("\nMean performance:")
display(df_results.mean().round(4))

print("\nStandard deviation:")
display(df_results.std().round(4))

print("\nBest iteration (per fold):", best_iterations)
print("Mean best iteration:", np.mean(best_iterations))


# %% [markdown]
# To better understand the impact of data scarcity on performance estimation, the results obtained with the original 70–15–15 train/validation/test split were re-evaluated using a more robust cross-validation strategy. In the initial fixed split, the test set contained only a small number of samples, which led to unstable estimates of generalisation performance and an overly optimistic test accuracy of approximately 0.89. Such fluctuations are expected in small datasets, where minor changes in sample composition can disproportionately affect the final metrics.
# 
# To address this limitation, a **5-fold stratified cross-validation** scheme with a 3–1–1 structure was adopted. In each fold, three partitions were used for training, one for validation (supporting early stopping), and one for testing. This design ensures that every sample is used for training, validation, and testing across different folds while preserving class balance. The resulting cross-validated estimate converged to a mean accuracy of 0.851 with a standard deviation of 0.036, demonstrating far greater statistical stability and reducing the uncertainty introduced by the small dataset size.
# 
# Whereas the original hold-out split produced a single optimistic estimate, the cross-validated approach revealed a more realistic and reliable measure of true generalisation behaviour. For this reason, the 3–1–1 cross-validated performance is adopted as the primary reference throughout the study. After establishing the robustness of the model via cross-validation, a final XGBoost model is trained using the full training portion of the data, with a separate 20% hold-out set reserved exclusively for fairness assessment and interpretability analysis.

# %% [markdown]
# #### 4.3.2 Gender-Specific Performance Analysis
# 
# Cross-validation was used exclusively to obtain a reliable estimate of the model’s generalisation performance under data scarcity. The 3–1–1 stratified cross-validation scheme provided a stable estimate by rotating training, validation, and testing subsets while preserving class balance. However, subgroup fairness analysis cannot rely on rotating folds, since the test partition changes at every iteration and does not allow consistent comparisons between demographic groups.
# Fairness and interpretability require a single, fixed model evaluated on a stable and unseen subset of the data. Therefore, after confirming the robustness of the XGBoost classifier through cross-validation, a final model was trained on the full 80% training portion of the dataset, using the same regularisation and early-stopping configuration validated earlier. The remaining 20% hold-out set, which is never used during training, was reserved exclusively for fairness assessment. This fixed hold-out subset ensures that predictions for male and female patients are generated under identical conditions, allowing reliable estimation of group-specific performance and outcome disparities.
# The following analysis evaluates whether the final XGBoost model behaves consistently across genders and whether systematic differences arise in accuracy, sensitivity, false-positive rates, or fairness metrics.

# %%
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

 
# 1. Prepare the target and features
 
# Binary target: 1 = disease present, 0 = no disease
y = (df["Heart Disease"] > 0).astype(int)

# Remove leakage columns from X
leak_cols = ["Heart Disease", "Heart Disease Num", "severity", "target", "num"]
X = df.drop(columns=[c for c in leak_cols if c in df.columns])

 
# 2. Train/Validation split for final model
#    (80% training, 20% internal validation)
 
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

 
# 3. Preprocess (fit only on training data)
 
X_train_proc = preprocessor.fit_transform(X_train_full)
X_val_proc   = preprocessor.transform(X_val_full)

# Convert to DMatrix
dtrain_full = xgb.DMatrix(X_train_proc, label=y_train_full)
dval_full   = xgb.DMatrix(X_val_proc,   label=y_val_full)

 
# 4. XGBoost hyperparameters (validated earlier)
 
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "seed": 42,
}

 
# 5. Train the final model with early stopping
 
booster_final = xgb.train(
    params=params,
    dtrain=dtrain_full,
    num_boost_round=1000,
    evals=[(dval_full, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

print("Final model trained. Best iteration:", booster_final.best_iteration)

 
# 6. Wrapper for sklearn-like interface
 
class XGBWrapper:
    def __init__(self, booster):
        self.model = booster
    
    def predict_proba(self, X):
        dm = xgb.DMatrix(X)
        prob = self.model.predict(dm)
        return np.vstack([1 - prob, prob]).T
    
    def predict(self, X):
        dm = xgb.DMatrix(X)
        prob = self.model.predict(dm)
        return (prob >= 0.5).astype(int)

# Final model to be used for fairness and interpretability
xgb_final = XGBWrapper(booster_final)

 
# 7. Leakage and split integrity checks
 
def check_split(name, X_split, y_split):
    print(f"\n=== {name} ===")
    print("Shape:", X_split.shape)
    print("Columns:", list(X_split.columns))
    
    # check if forbidden columns are still present
    forbidden = {"Heart Disease", "Heart Disease Num", "severity", "target", "num"}
    present_forbidden = forbidden.intersection(set(X_split.columns))
    if present_forbidden:
        print("WARNING: Forbidden leakage columns detected:", present_forbidden)
    else:
        print("OK: No leakage columns detected.")

    # check y distribution
    print("y distribution:", y_split.value_counts().to_dict())

# call checks
check_split("TRAIN FULL", X_train_full, y_train_full)
check_split("VALIDATION FULL", X_val_full, y_val_full)




# %% [markdown]
# ##### 4.3.2.1 Split the test set by gender

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Preprocess test data
X_test_proc = preprocessor.transform(X_test)

 
# 2. Split test set by gender
 
sex_test = X_test["sex"].values

X_test_male   = X_test_proc[sex_test == 1]
y_test_male   = y_test[sex_test == 1]

X_test_female = X_test_proc[sex_test == 0]
y_test_female = y_test[sex_test == 0]


# %% [markdown]
# ##### 4.3.2.2 Generate gender-specific predictions

# %%
y_pred_male   = xgb_final.predict(X_test_male)
y_pred_female = xgb_final.predict(X_test_female)

y_prob_male   = xgb_final.predict_proba(X_test_male)[:, 1]
y_prob_female = xgb_final.predict_proba(X_test_female)[:, 1]

# %% [markdown]
# ##### 4.3.2.3 Compute standard performance metrics

# %%
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "Accuracy": acc,
        "F1": f1,
        "TPR (Sensitivity)": tpr,
        "FPR": fpr,
        "Support": len(y_true)
    }

metrics_male   = compute_metrics(y_test_male,   y_pred_male)
metrics_female = compute_metrics(y_test_female, y_pred_female)

gender_summary = pd.DataFrame(
    [ {"Group": "Male", **metrics_male},
      {"Group": "Female", **metrics_female} ]
)

display(gender_summary.round(4))


# %% [markdown]
# ##### 4.3.2.4 Compute fairness metrics

# %%
def fairness_metrics(y_male, y_female, y_pred_male, y_pred_female):
    
    # Positive prediction rates
    sr_male   = y_pred_male.mean()
    sr_female = y_pred_female.mean()

    # Statistical Parity
    SP = sr_female - sr_male

    # Disparate Impact Ratio
    DIR = sr_female / sr_male if sr_male > 0 else np.nan

    # Confusion matrices
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_male, y_pred_male).ravel()
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_female, y_pred_female).ravel()

    # True Positive Rate difference
    TPR_m = tp_m / (tp_m + fn_m)
    TPR_f = tp_f / (tp_f + fn_f)
    EO_gap = TPR_f - TPR_m

    # Equalized Odds Difference
    FPR_m = fp_m / (fp_m + tn_m)
    FPR_f = fp_f / (fp_f + tn_f)
    AOD = 0.5 * ((FPR_f - FPR_m) + (TPR_f - TPR_m))

    return {
        "Statistical Parity": SP,
        "Disparate Impact Ratio": DIR,
        "Equal Opportunity Gap": EO_gap,
        "Average Odds Difference": AOD
    }

fairness_results = fairness_metrics(
    y_test_male, y_test_female,
    y_pred_male, y_pred_female
)

fairness_df = pd.DataFrame([fairness_results])

display(fairness_df.round(4))

# %% [markdown]
# The male subgroup exhibits higher sensitivity (TPR = 0.9524) than the female subgroup (TPR = 0.7143), indicating that the classifier detects heart disease more reliably among male patients. However, the female subgroup shows a lower false positive rate (FPR = 0.0 versus 0.20), meaning that the model avoids incorrect positive predictions more effectively for women. Accuracy remains comparable across groups (0.878 for men and 0.900 for women), but F1-scores reveal asymmetry in predictive balance: males display a higher F1 (0.8889), while females show a reduction (0.8333), consistent with their lower sensitivity.
# 
# Fairness indicators reveal disparities in predicted positive rates. Statistical parity is negative (−0.3354), and the disparate impact ratio is substantially below the commonly referenced 0.8 threshold (0.4271), indicating that women receive positive predictions at a markedly lower rate. The equal opportunity gap (−0.2381) confirms that the lower positive prediction rate among women reflects a lower true positive rate rather than a difference in false positives. The average odds difference (−0.219) reinforces that both TPR and FPR deviations contribute to this imbalance.
# 
# These results indicate that, although overall classification performance is high in both groups, the model’s decision boundary is not applied uniformly across genders. The classifier is more sensitive but less specific for male patients, while for female patients it is more conservative, producing fewer false positives but missing a larger fraction of true cases. Such patterns are consistent with known clinical differences in symptom presentation yet highlight the risk of unequal error distribution in automated diagnostic support tools.

# %% [markdown]
# ##### 4.3.2.5 Fairness Mitigation via Threshold Adjustment

# %%

thr_male = 0.50
thr_female = 0.40   # lowered threshold to increase sensitivity for women

# New predictions with group-specific thresholds
y_pred_male_adj = (y_prob_male >= thr_male).astype(int)
y_pred_female_adj = (y_prob_female >= thr_female).astype(int)

# Recompute metrics
metrics_male_adj   = compute_metrics(y_test_male,   y_pred_male_adj)
metrics_female_adj = compute_metrics(y_test_female, y_pred_female_adj)

gender_summary_adj = pd.DataFrame(
    [ {"Group": "Male (adj)", **metrics_male_adj},
      {"Group": "Female (adj)", **metrics_female_adj} ]
)

display(gender_summary_adj.round(4))

# Fairness metrics after mitigation
fairness_results_adj = fairness_metrics(
    y_test_male, y_test_female,
    y_pred_male_adj, y_pred_female_adj
)

fairness_df_adj = pd.DataFrame([fairness_results_adj])
display(fairness_df_adj.round(4))


# %% [markdown]
# A simple post-processing intervention was applied to reduce the gender disparity previously observed in sensitivity. The model architecture and training remained unchanged; only the decision threshold was modified, keeping 0.50 for male patients and lowering it to 0.40 for female patients.
# 
# This adjustment substantially improved female sensitivity (from 0.7143 to 0.8571) and increased their F1-score, while preserving overall accuracy and not introducing false positives in this subgroup. Male performance remained effectively stable. Fairness metrics reflect this improvement: the Equal Opportunity Gap decreased in magnitude and the Disparate Impact Ratio increased, although both still indicate residual disparity. Statistical Parity and Average Odds Difference also improved but were not fully corrected.
# 
# In summary, threshold adjustment effectively mitigated the most critical disparity—under-detection in women—illustrating how post-processing can partially improve group fairness without retraining the model, while also showing that deeper structural imbalances require more advanced mitigation strategies.

# %% [markdown]
# #### 4.3.3 Chest-Pain–Specific Performance Analysis
# 
# Chest pain type is a clinically important feature that reflects different phenotypic presentations of cardiovascular disease. Because each category carries distinct symptom patterns and risk profiles, it is necessary to verify whether the classifier behaves consistently across these subgroups. Following the same logic as the gender-based analysis, predictions on the fixed 20% hold-out set were examined separately for each chest-pain category, focusing on accuracy, sensitivity, F1-score, and false-positive rate. 
# 
# This assessment reveals whether the model treats certain symptom profiles differently, with particular attention to asymptomatic and non-anginal cases, which are clinically harder to diagnose and more prone to misclassification.

# %%
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

def compute_cp_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    TPR = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # Sensitivity
    FNR = fn / (tp + fn) if (tp + fn) > 0 else np.nan   # Miss rate
    FPR = fp / (fp + tn) if (fp + tn) > 0 else np.nan   # False positive rate

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TPR (Sensitivity)": TPR,
        "FNR (Miss Rate)": FNR,
        "FPR": FPR,
        "Support": len(y_true)
    }

results_cp = []

# Loop over chest-pain categories (1, 2, 3, 4)
for cp_value in sorted(X_test["cp"].unique()):
    mask = X_test["cp"] == cp_value
    
    X_sub = X_test[mask]
    y_sub = y_test[mask]

    if len(y_sub) == 0:
        continue

    # Predictions for this CP type
    y_pred = xgb_final.predict(X_sub)

    metrics = compute_cp_metrics(y_sub, y_pred)
    metrics["ChestPainType"] = cp_value

    results_cp.append(metrics)

# Convert to DataFrame for presentation
df_cp = pd.DataFrame(results_cp)[[
    "ChestPainType", "Support", "Accuracy", "F1",
    "TPR (Sensitivity)", "FNR (Miss Rate)", "FPR"
]]

display(df_cp.round(4))

# %% [markdown]
# The model exhibits heterogeneous behaviour across chest-pain categories. For typical (CP=1) and atypical angina (CP=2), sensitivity reaches 1.00, indicating that all true disease cases are detected; however, this occurs alongside high false-positive rates, suggesting an overly aggressive decision boundary for these symptom profiles. In contrast, performance declines sharply for non-anginal pain (CP=3), where sensitivity drops to 0.33 and the miss rate increases substantially, reflecting the clinical difficulty of identifying disease in patients with atypical presentations. Asymptomatic cases (CP=4) show a more balanced pattern, with moderate sensitivity and no false positives. Overall, the classifier tends to over-predict disease in classic symptom categories and under-predict in ambiguous or non-anginal presentations, illustrating the need for phenotype-aware evaluation in clinical risk models.

# %%
# Column containing original severity levels (0–4)
severity_col = "severity"

df_results = []

# Loop over severity classes present in the test set
for sev in sorted(df.loc[X_test.index, severity_col].unique()):
    
    mask = df.loc[X_test.index, severity_col] == sev
    
    X_sub = X_test[mask]
    y_sub = y_test[mask]   # binary target (0/1)

    if len(y_sub) == 0:
        continue
    
    # Predictions
    y_pred = xgb_final.predict(X_sub)
    
    # Confusion matrix for binary classifier
    tn, fp, fn, tp = confusion_matrix(y_sub, y_pred, labels=[0,1]).ravel()
    
    TPR = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    FNR = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    FPR = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    df_results.append({
        "Severity": sev,
        "Support": len(y_sub),
        "Accuracy": accuracy_score(y_sub, y_pred),
        "F1": f1_score(y_sub, y_pred, zero_division=0),
        "TPR": TPR,
        "FNR": FNR,
        "FPR": FPR
    })

df_severity = pd.DataFrame(df_results)
df_severity


# %% [markdown]
# The severity-based analysis evaluates whether the binary classifier behaves consistently across the original angiographic disease levels (0–4). Since the model was trained only to distinguish presence versus absence of disease, this evaluation does not test its ability to discriminate between severity stages, but rather whether performance varies across clinically meaningful subgroups.
# Severity 0 contains only healthy patients, which naturally produces asymmetric metrics: the model achieves 72.7% accuracy but an F1-score of 0.0 because no positive cases exist in this group. For severities 1–3, accuracy ranges from 50% to 62.5%, with sensitivities between 0.50 and 0.63. These moderate values reflect the clinical difficulty of detecting mild or borderline disease. Severity 4 shows perfect scores, but this result is not reliable due to the extremely small support (n = 2).
# Overall, the model performs better for more clearly expressed disease and struggles with mild cases, a pattern consistent with clinical expectations and with the limited sample size available in each subgroup.

# %% [markdown]
# ##### 4.3.4 Interaction: Chest Pain × Sex Performance Analysis:

# %%
results = []

for cp in sorted(X_test['cp'].unique()):
    for sex_val, sex_label in [(0, "Female"), (1, "Male")]:
        
        mask = (X_test['cp'] == cp) & (X_test['sex'] == sex_val)
        
        if mask.sum() == 0:
            continue
        
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        
        y_pred = xgb_final.predict(X_sub)
        
        tn, fp, fn, tp = confusion_matrix(y_sub, y_pred, labels=[0,1]).ravel()
        
        TPR = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        FPR = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        FNR = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        
        results.append({
            "ChestPainType": cp,
            "Sex": sex_label,
            "Support": len(y_sub),
            "Accuracy": accuracy_score(y_sub, y_pred),
            "F1": f1_score(y_sub, y_pred, zero_division=0),
            "TPR": TPR,
            "FNR": FNR,
            "FPR": FPR
        })

df_interaction = pd.DataFrame(results)
df_interaction


# %% [markdown]
# The interaction analysis between chest-pain type and sex reveals heterogeneous model behaviour across subgroups. Performance is generally stable for male patients, with relatively high sensitivity across all chest-pain types (except non-anginal pain, where the miss rate increases). In contrast, the model shows reduced reliability for female patients, particularly for atypical angina and non-anginal pain, where the classifier systematically fails to detect positive cases (TPR = 0 and F1 = 0). These patterns reflect well-known clinical asymmetries: women frequently present non-typical symptoms, which leads to greater diagnostic ambiguity and appears mirrored in the model’s decision boundary. For asymptomatic and typical-angina presentations, sensitivity improves in both sexes, although sample sizes remain small. Overall, the model demonstrates unequal error distribution across sex × phenotype subgroups, with the weakest performance concentrated in female patients presenting atypical or non-anginal symptoms.

# %% [markdown]
# ### 4.4 SHAP-Based Explainability
# 
# To complement the performance and fairness analyses presented in the previous sections, we employ SHAP (SHapley Additive exPlanations) to investigate how the XGBoost classifier assigns importance to each clinical feature. SHAP provides a mathematically grounded decomposition of each prediction into additive feature contributions, allowing both global and subgroup-specific interpretability. This analysis supports two central objectives of the study: understanding the decision logic of the model and determining whether the same clinical signals are used uniformly across demographic and phenotypic groups.
# The explainability analysis proceeds in three stages. First, global SHAP values are computed to summarise the overall influence of each feature on the prediction function. Next, the same methodology is applied separately to male and female patients, enabling the detection of potential divergences in how the model processes clinical symptoms across genders. Finally, SHAP distributions are examined across chest-pain categories to evaluate whether the model’s reliance on individual features changes across different clinical phenotypes. Together, these analyses provide a transparent, model-agnostic view of the classifier’s behaviour and help identify whether subgroup disparities arise from underlying differences in feature attribution.

# %% [markdown]
# #### 4.4.1 Global SHAP
# 
# Global SHAP values were computed on the fixed 20% hold-out set using the TreeExplainer, which provides exact Shapley attributions for tree-based models. The resulting importance distribution captures the average magnitude and direction of each feature’s contribution to the predicted probability of heart disease. Features related to cardiac workload and vascular obstruction—such as ST-segment depression (oldpeak), maximum heart rate (thalach), and the number of major vessels (ca)—exhibit the strongest influence on model outputs, consistent with established clinical determinants of ischemic risk. Binary attributes such as exercise-induced angina (exang) and thalassemia patterns also contribute meaningfully to the prediction function, while demographic variables such as age and sex present comparatively smaller effects.
# Overall, the global SHAP profile indicates that the model relies predominantly on physiologically grounded predictors and that the decision boundary is shaped by a clinically coherent set of features. This global interpretability step establishes a reference against which subgroup-specific attributions—by gender and by chest-pain phenotype—can later be compared to assess whether the model applies its decision rules consistently across all patient groups.

# %%
leak_cols = ['Heart Disease', 'Heart Disease Num', 'severity', 'target', "num"]
X = df.drop(columns=[c for c in leak_cols if c in df.columns])
y = df['target']

# %%
import shap
import matplotlib.pyplot as plt

feature_names = preprocessor.get_feature_names_out()
explainer = shap.TreeExplainer(booster_final)
shap_values = explainer.shap_values(X_test_proc)

# Criar figura antes do SHAP
plt.figure(figsize=(4, 5))

shap.summary_plot(
    shap_values,
    X_test_proc,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.show()



# %% [markdown]
# Numeric variables are scaled (num__ prefix), while categorical variables are one-hot encoded (cat__).

# %% [markdown]
# The SHAP summary plot shows that the model’s predictions are driven mainly by a small set of clinically relevant features. The number of major vessels (*ca*) is by far the strongest contributor: higher values consistently push the prediction toward “disease.” Certain chest-pain encodings, especially the asymptomatic category (*cp=4*), also exert a substantial effect, indicating that symptom presentation strongly shapes the decision boundary. Thalassemia indicators (*thal*) appear as additional key drivers, while age, cholesterol, *oldpeak*, and *thalach* contribute more moderately but remain directionally consistent with known clinical risk factors. By contrast, resting blood pressure, ECG categories, and exercise-induced angina show limited influence. Overall, the model relies most heavily on structural cardiac markers and specific symptom patterns, with demographic signals playing a comparatively minor role.
# 

# %% [markdown]
# #### 4.4.2 SHAP by Gender
# 
# To evaluate whether the model relies on clinical features in a gender-consistent way, SHAP values were computed separately for male and female patients. This subgroup analysis follows the group-fairness framing used previously (Module 04), allowing us to examine whether discrepancies in performance metrics (e.g., TPR and FPR gaps) correspond to divergences in the underlying attribution patterns. By comparing SHAP distributions across genders, we identify potential shifts in the model’s reliance on physiological signals, symptom encodings, or demographic proxies that may contribute to unequal error allocation.
# 
# 

# %%
import matplotlib.pyplot as plt
import shap
import numpy as np

sex_test = X_test["sex"].values
male_idx   = (sex_test == 1)
female_idx = (sex_test == 0)

# Keep all features (including sex)
X_test_proc_male   = X_test_proc[male_idx]
X_test_proc_female = X_test_proc[female_idx]

shap_values_male   = shap_values[male_idx]
shap_values_female = shap_values[female_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Male
plt.sca(axes[0])
shap.summary_plot(
    shap_values_male,
    X_test_proc_male,
    feature_names=feature_names,
    show=False
)
axes[0].set_title("SHAP Summary — Male")

# Female
plt.sca(axes[1])
shap.summary_plot(
    shap_values_female,
    X_test_proc_female,
    feature_names=feature_names,
    show=False
)
axes[1].set_title("SHAP Summary — Female")

plt.tight_layout()
plt.show()


# %% [markdown]
# The resulting SHAP distributions show that both groups rely on the same core physiological predictors—most notably the number of major vessels (`ca`), ST-segment depression (`oldpeak`), and thalassemia encodings. The asymptomatic chest-pain category (`cp=4`) also contributes meaningfully, although its relative influence differs across groups. Importantly, the magnitude and dispersion of SHAP values vary substantially between genders.
# 
# Male patients exhibit stronger and more concentrated positive contributions for `ca`, `oldpeak`, and high-risk `thal` categories, indicating that the model responds more decisively to structural markers of ischemia in this group. This behaviour aligns with their higher sensitivity (TPR) observed in the performance analysis.
# 
# Female patients display more diffuse attributions, with attenuated SHAP magnitudes for the strongest cardiac indicators and proportionally higher influence from discrete symptom encodings and secondary risk markers. This pattern reflects a more conservative decision boundary for women and is consistent with their reduced sensitivity and the fairness gaps quantified in Section 4.3.
# 
# Overall, SHAP provides a mechanistic explanation for the gender disparities previously identified: the model assigns weaker importance to obstructive cardiac features in women, making positive predictions harder to trigger and contributing to lower true-positive rates in this subgroup.
# 

# %% [markdown]
# #### 4.4.3. SHAP by Chest Pain Type
# To determine whether these disparities stem from differences in how the model internally processes each phenotype, SHAP values were computed separately for each CP subgroup.
# 

# %%
import matplotlib.pyplot as plt
import numpy as np

# Mapping from CP code → clinical name
cp_labels = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-Anginal Pain",
    4: "Asymptomatic"
}

cp_values = sorted(X_test["cp"].unique())
n_cp = len(cp_values)

# Use a clean 2×2 layout (28×14 inches)
fig, axes = plt.subplots(2, 2, figsize=(28, 14))
axes = axes.flatten()

for i, cp in enumerate(cp_values):
    mask = (X_test["cp"] == cp)

    X_cp_proc = X_test_proc[mask]
    shap_cp   = shap_values[mask]

    plt.sca(axes[i])
    shap.summary_plot(
        shap_cp,
        X_cp_proc,
        feature_names=feature_names,
        show=False,
        plot_size=None
    )

    title = f"SHAP Summary — {cp_labels[cp]} (CP = {cp})"
    axes[i].set_title(title, fontsize=18)

# manual layout to avoid overlap
plt.subplots_adjust(
    left=0.03, right=0.97,
    top=0.90, bottom=0.08,
    wspace=0.30, hspace=0.30
)

plt.show()


# %%
import numpy as np
import pandas as pd

sex0_idx = list(feature_names).index("cat__sex_0.0")
sex1_idx = list(feature_names).index("cat__sex_1.0")

results = []

for cp in sorted(X_test["cp"].unique()):
    mask = (X_test["cp"] == cp)
    
    shap_cp = shap_values[mask]
    sex_vals = X_test["sex"][mask].values

    # SHAP effect only for individuals who are sex = 0 (female)
    shap_sex0 = shap_cp[sex_vals == 0, sex0_idx]

    # SHAP effect only for individuals who are sex = 1 (male)
    shap_sex1 = shap_cp[sex_vals == 1, sex1_idx]

    results.append({
        "CP": cp,
        "Chest Pain Type": {1:"Typical",2:"Atypical",3:"Non-Anginal",4:"Asympt"}[cp],
        "Mean |SHAP|(sex=0)": np.mean(np.abs(shap_sex0)) if len(shap_sex0)>0 else np.nan,
        "Mean |SHAP|(sex=1)": np.mean(np.abs(shap_sex1)) if len(shap_sex1)>0 else np.nan,
        "Support sex=0": len(shap_sex0),
        "Support sex=1": len(shap_sex1)
    })

df_sex_by_cp = pd.DataFrame(results)
df_sex_by_cp


# %% [markdown]
# Chest pain type is a key clinical indicator, and analysing SHAP values per subgroup clarifies how the model adapts its decision logic to different symptom profiles. Across all chest-pain types, the same physiological markers (ca, oldpeak, thal, thalach) remain the dominant predictors, but their influence varies considerably.
# 
# For typical and atypical angina (CP=1–2), these features show large and stable SHAP contributions, indicating high model confidence and consistent alignment with classical ischemic patterns—matching the high sensitivities observed earlier. In non-anginal pain (CP=3), the SHAP structure becomes weaker and more diffuse: core markers lose strength, and categorical variables gain relative importance. This reflects greater diagnostic uncertainty and explains the substantial drop in sensitivity for this subgroup. Asymptomatic patients (CP=4) again rely strongly on objective markers, but with higher variability across individuals.
# 
# The contribution of sex, however, does not vary substantially across chest-pain categories. SHAP values show a systematic and asymmetric pattern: being female (sex=0) consistently produces a strong negative contribution, whereas being male (sex=1) produces a small positive one. **This means that sex influences the prediction in a similar way for all symptom profiles**. Importantly, this structural effect suppresses the probability of disease for women regardless of chest-pain type, helping to explain their lower sensitivity observed in the fairness analysis. Rather than depending on phenotype, the impact of sex reflects a global adjustment learned by the model from the dataset’s demographic distribution.
# 

# %% [markdown]
# #### 4.4.4. Local Explanations
# 
# While the global and subgroup-level SHAP analyses clarify the overall decision logic of the model, they do not reveal how individual predictions succeed or fail in practice. To investigate these mechanisms at the instance level, we now turn to local explanations.
# 

# %% [markdown]
# Local explanations allow us to inspect the model’s behaviour on individual cases and to identify systematic patterns behind correct and incorrect predictions. Using SHAP waterfall plots for representative instances, we examine how specific feature contributions push each prediction toward or away from the positive class. This perspective is essential for understanding whether misclassifications arise from weak clinical evidence, conflicting features, or structural biases identified in earlier analyses.
# 
# Correctly classified positive cases typically show a coherent combination of strong physiological indicators—such as high ca, elevated oldpeak, and abnormal thal results—whose SHAP contributions accumulate decisively toward a disease prediction. These cases act as “prototypes” for the model: they align well with patterns learned during training and illustrate how the classifier recognises clear ischemic signatures.
# 
# Incorrect predictions, however, reveal recurring failure modes. False negatives often correspond to borderline presentations where physiological markers are present but do not exceed the decision threshold, resulting in a muted SHAP profile with dispersed contributions. This is particularly visible in women with non-anginal symptoms, where negative SHAP contributions associated with sex suppress the overall probability of disease. False positives, by contrast, frequently arise from noisy or isolated features—such as a mild elevation in oldpeak or categorical ECG abnormalities—that the model overweights in the absence of stronger counterbalancing evidence. These instances behave as “outliers,” where the local explanation highlights a single dominant feature driving an incorrect decision.
# 
# Overall, local SHAP inspection demonstrates that the model’s errors are not random: they systematically reflect clinical ambiguity, feature imbalance, and demographic asymmetries. By identifying these patterns at the instance level, local explanations complement the global analysis and provide actionable insights for refining the model or adjusting decision thresholds in deployed clinical settings.
# 

# %%
import matplotlib.pyplot as plt

# Masks
y_pred = xgb_final.predict(X_test_proc)

FN_mask = (y_test == 1) & (y_pred == 0)   # False Negatives
FP_mask = (y_test == 0) & (y_pred == 1)   # False Positives

shap_FN = shap_values[FN_mask]
shap_FP = shap_values[FP_mask]

X_FN = X_test_proc[FN_mask]
X_FP = X_test_proc[FP_mask]

# Create side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# --- FN plot ---
plt.sca(axes[0])
shap.summary_plot(
    shap_FN,
    X_FN,
    feature_names=feature_names,
    show=False,
    plot_size=None
)
axes[0].set_title("SHAP Summary – False Negatives", fontsize=16)

# --- FP plot ---
plt.sca(axes[1])
shap.summary_plot(
    shap_FP,
    X_FP,
    feature_names=feature_names,
    show=False,
    plot_size=None
)
axes[1].set_title("SHAP Summary – False Positives", fontsize=16)

plt.subplots_adjust(
    left=0.03, right=0.97,
    bottom=0.10, top=0.90,
    wspace=0.35
)

plt.show()


# %% [markdown]
# **Identifying the Most Unstable Misclassifications Using SHAP-Based Instability Scores**

# %%
instability = np.sum(np.abs(shap_values), axis=1)

df_errors = pd.DataFrame({
    "true": y_test,
    "pred": y_pred,
    "instability": instability
})

df_errors = df_errors[(df_errors.true != df_errors.pred)].sort_values("instability", ascending=False)
df_errors.head()


# %% [markdown]
# **Ranking Feature Contributions for False Negatives and False Positives**

# %%
mean_FN = np.mean(np.abs(shap_FN), axis=0)
mean_FP = np.mean(np.abs(shap_FP), axis=0)

pd.DataFrame({
    "feature": feature_names,
    "FN_importance": mean_FN,
    "FP_importance": mean_FP
}).sort_values("FN_importance", ascending=False)


# %% [markdown]
# The code computes the mean absolute SHAP value for each feature across all false negatives (FN) and false positives (FP), producing two importance profiles that highlight which variables most strongly drive each type of error.

# %% [markdown]
# To better understand how the model fails, SHAP values were analysed separately for false negatives (FN: true=1, pred=0) and false positives (FP: true=0, pred=1). The side-by-side SHAP summary plots and the feature-level ranking of mean absolute SHAP values reveal that the same core variables (number of major vessels (ca), asymptomatic chest pain (cp=4), age, thalassemia patterns (thal), and sex) are also dominant drivers of misclassifications.
# 
# For false negatives, the SHAP summary plot shows that low values of ca (blue points with negative SHAP) and the female indicator (sex=0) often exert strong negative contributions, pulling the prediction towards the “no disease” class. At the same time, some positive evidence is present, such as asymptomatic presentations (cp=4) or abnormal thal results, which contribute positively to the logit. However, these positive contributions are frequently offset by the negative effects of ca, sex=0, and other features, so that the final prediction remains below the decision threshold. This pattern is consistent with the feature ranking, where ca, cp=4, age, thal=3 and sex=0 have the highest mean |SHAP| among false negatives, indicating that missed cases tend to occur in older patients with partially expressed risk factors whose overall evidence is weakened by low ca and the downward adjustment associated with being female.
# 
# False positives display a complementary structure. In this group, high values of ca, cp=4, oldpeak and thal=3.0 generate large positive SHAP contributions, pushing the model towards the “disease” class even though the ground truth label is 0. The mean |SHAP| scores confirm that ca, cp=4, oldpeak and thal remain the most influential features for false positives, with restecg and slope also gaining importance. These errors therefore arise primarily in true negatives who nonetheless exhibit strong or atypical patterns in the core risk markers used by the model.
# 
# To systematically locate complex misclassifications, an instability score was computed as the sum of absolute SHAP values across all features for each instance. The highest-instability cases include both false negatives and false positives, and are characterised by multiple features with large, sometimes competing contributions. These instances are particularly informative for local inspection using waterfall plots, as they expose how the interplay between ca, chest pain, sex and other predictors can lead the model to confident yet incorrect decisions.
# 

# %% [markdown]
# **Local SHAP Waterfall Explanations for Representative Correct and Incorrect Predictions**

# %% [markdown]
# To complement the global and subgroup-level SHAP analyses, we examine local explanations for four representative instances from the test set: one true positive (TP), one true negative (TN), the most unstable false negative (FN), and the most unstable false positive (FP). These cases were selected to illustrate how the model assembles evidence at the individual level, and to clarify the mechanisms behind both reliable and unreliable predictions. For each instance, a SHAP waterfall plot was generated, decomposing the predicted log-odds into additive feature contributions relative to the model’s expected value.

# %%
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Identify TP, TN, FN, FP
all_idx = np.arange(len(y_test))

TP_mask = (y_test == 1) & (y_pred == 1)
TN_mask = (y_test == 0) & (y_pred == 0)
FN_mask = (y_test == 1) & (y_pred == 0)
FP_mask = (y_test == 0) & (y_pred == 1)

tp_idx = all_idx[TP_mask]
tn_idx = all_idx[TN_mask]
fn_idx = all_idx[FN_mask]
fp_idx = all_idx[FP_mask]

i_tp = tp_idx[0]
i_tn = tn_idx[0]

# 2. Most unstable FN/FP
worst_fn_id = df_errors[(df_errors["true"] == 1) & (df_errors["pred"] == 0)].index[0]
worst_fp_id = df_errors[(df_errors["true"] == 0) & (df_errors["pred"] == 1)].index[0]

pos_fn_worst = np.where(X_test.index == worst_fn_id)[0][0]
pos_fp_worst = np.where(X_test.index == worst_fp_id)[0][0]

# 3. Legacy SHAP waterfall helper
def legacy_waterfall(shap_values_row, title):
    plt.title(title, fontsize=18)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_row,
            base_values=explainer.expected_value,
            feature_names=feature_names
        ),
        max_display=12,
        show=False
    )

# -------------------------------
# 4. Improved Layout for Clarity
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(50, 28))   # MUCH larger and wider
axes = axes.flatten()

# Set same x-range across all plots
X_MIN, X_MAX = -2.5, 3.5
for ax in axes:
    ax.set_xlim(X_MIN, X_MAX)

# Draw each SHAP waterfall inside its axis
plt.sca(axes[0])
legacy_waterfall(shap_values[i_tp], "TP Example")

plt.sca(axes[1])
legacy_waterfall(shap_values[i_tn], "TN Example")

plt.sca(axes[2])
legacy_waterfall(shap_values[pos_fn_worst], "Most Unstable FN")

plt.sca(axes[3])
legacy_waterfall(shap_values[pos_fp_worst], "Most Unstable FP")

# Better spacing
plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05,
                    wspace=0.38, hspace=0.53)

plt.show()


# %% [markdown]
# 
# The TP and TN explanations reveal a highly coherent decision process. In the TP example, the model assigns strong positive contributions to canonical markers of cardiac pathology (namely the number of major vessels (*ca*), asymptomatic chest pain (*cp=4*), ST-segment depression (*oldpeak*), and abnormal thalassemia encodings). These features collectively push the prediction confidently toward the disease class, mirroring established clinical patterns. In the TN example, the absence or low magnitude of these markers produces predominantly negative contributions, consistently lowering the predicted risk. These cases demonstrate that the model behaves reliably when the clinical evidence is either strongly supportive of disease or strongly indicative of health.
# 
# In contrast, the most unstable FN and FP cases expose the model’s vulnerabilities. The FN instance exhibits a mixture of weak or borderline indicators: *ca*, *oldpeak*, and *thal* values contribute insufficiently to surpass the decision threshold despite representing a true disease case. This aligns with earlier subgroup analyses showing reduced sensitivity for presentations with subtle or non-classical features. The FP instance, on the other hand, illustrates over-reliance on a small set of strong but isolated predictors—most notably *ca* and *cp=4*—which leads the model to overestimate risk in the absence of broader supporting evidence.
# 
# Together, these local explanations reveal a consistent pattern: the model is robust when confronted with clear and clinically aligned signals, but it becomes unstable when evidence is incomplete, ambiguous, or strongly unbalanced. Errors arise primarily from cases where key features are either too weak (leading to false negatives) or disproportionately influential (leading to false positives). These findings reinforce the importance of integrating richer clinical context and potentially rebalancing feature contributions to improve reliability in borderline and atypical presentations.

# %% [markdown]
# **SHAP Dependence Plot for `num__ca`**

# %%
shap.dependence_plot("num__ca", shap_values, X_test_proc, feature_names=feature_names)


# %% [markdown]
# 
# The dependence plot for `num__ca` reveals a strongly monotonic relationship between the number of major vessels affected and the model’s predicted risk of heart disease. SHAP values increase sharply as `ca` rises from 0 to 3, indicating that this variable exerts one of the strongest positive influences on the decision function. The colour overlay suggests only minimal interaction with `restecg`, with points of different colours following nearly identical SHAP trajectories. This confirms that the model treats `ca` as a dominant, largely independent structural marker of disease severity, consistent with established clinical reasoning.

# %% [markdown]
# ### 6. Ethical Reflection and Discussion

# %% [markdown]
# #### 6.1 Phenotypic and gender bias in heart disease prediction
# 
# The empirical results of this study confirm that model performance is not uniform across patient subgroups. Sensitivity is substantially lower for patients with non-anginal chest pain and for women, even when overall test performance is reasonably high. These disparities reflect and potentially reinforce well-documented clinical patterns of underdiagnosis in female patients and in individuals who present with less “typical” ischaemic symptoms.
# 
# From an ethical perspective, these findings raise concerns about distributive justice and non-discrimination. A classifier that systematically produces more false negatives for specific subgroups implicitly encodes a lower standard of protection for those patients. In practice, this may translate into delayed diagnosis, reduced access to timely treatment, and a higher burden of preventable morbidity for individuals whose symptoms deviate from canonical male-centric presentations of coronary artery disease.
# 
# Importantly, the SHAP analyses show that sex and chest-pain phenotype are not merely passive descriptors but actively shape the decision boundary. Being female consistently exerts a strong negative contribution on the predicted probability of disease, while being male has only a small positive effect, regardless of chest-pain type. This structural pattern suggests that the model has learned to down-weight the risk associated with female patients in a global way, which helps explain the reduced sensitivity observed for women across symptom profiles. Such behaviour exemplifies phenotypic and demographic bias: the model inherits and amplifies imbalances present in the training data, instead of correcting for them.
# 

# %% [markdown]
# 
# #### 6.2 Potential harms and societal impact
# 
# If deployed in a clinical workflow, these biases would not be purely technical artefacts. They would materialise as tangible harms for particular groups of patients. False negatives in women or in patients with non-anginal pain could reinforce existing inequities in cardiovascular care, by making it less likely that these individuals are referred for further diagnostic testing or receive aggressive treatment. This risk is especially problematic in cardiology, where delays of hours or days can significantly change prognosis.
# 
# At a societal level, such patterns undermine public trust in AI-assisted medicine. If patients or clinicians perceive that decision-support tools tend to “miss” disease in certain populations, this may foster justified scepticism towards algorithmic systems in healthcare more broadly. Moreover, the opacity of complex models can make it difficult for patients to understand why they were not flagged as high-risk, raising concerns about transparency, accountability, and the right to an explanation in medical decision-making.
# 
# The unequal distribution of errors also interacts with broader structural inequities. Groups that have historically faced underrepresentation in clinical studies or systematic dismissal of their symptoms are precisely those who may be further disadvantaged by biased models. Ethical evaluation therefore cannot be separated from questions of data provenance, representativeness, and the socio-historical context in which the dataset was collected.
# 

# %% [markdown]
# 
# #### 6.3 Relation to the EU AI Act and regulatory classification
# 
# Under the EU AI Act, this type of predictive model qualifies as an AI system because it uses machine-learning techniques to generate predictions that support decision-making in a specific environment (cardiovascular diagnosis). If integrated into a diagnostic or triage pipeline, it would likely fall under the category of high-risk medical AI systems, given its potential impact on patients’ health and fundamental rights.
# 
# In a high-risk scenario, both developers (“providers”) and clinical institutions (“deployers”) would face specific legal obligations. Providers would need to implement a documented risk-management system, ensure appropriate data governance and quality, and produce technical documentation that describes the model, its intended purpose, performance metrics, and known limitations. Deployers would be responsible for understanding these limitations, monitoring real-world performance, and avoiding over-reliance on algorithmic outputs in the absence of clinical judgement.
# 
# Transparency and explainability are explicitly highlighted in the EU AI Act as essential requirements for high-risk AI. The use of SHAP in this study partially anticipates such obligations by making feature contributions interpretable and by revealing systematic biases against certain subgroups. However, in a real deployment, explainability would need to be complemented with clear documentation for clinicians, user-friendly interfaces for inspecting explanations, and institutional procedures for acting on detected biases (e.g., revising thresholds, retraining with improved data, or suspending use if harmful patterns are identified).
# 
# It is also important to emphasise that, in its current form, the model is a research artefact trained on a historical dataset and not a certified medical device. The AI Act does not directly apply to this academic prototype. Nevertheless, using its framework as a reference helps identify which safeguards would be necessary before any clinical integration.
# 

# %% [markdown]
# 
# #### 6.4 Responsible AI practices implemented in this study
# 
# The analysis already incorporates several responsible-AI practices that align with ethical and regulatory expectations:
# 
# - A Data-Centric AI perspective was adopted from the start, with careful exploration of class balance, subgroup distributions, and potential sources of phenotypic bias, rather than focusing exclusively on algorithmic complexity.
# - Fairness was explicitly evaluated by comparing performance across sex and chest-pain subgroups, instead of reporting only global metrics. This made visible the trade-off between aggregate performance and subgroup-level harms.
# - Model behaviour was probed using SHAP, both globally and locally, to understand which features drive predictions and how they interact in typical and atypical cases. This provided concrete evidence of structural bias (e.g., the systematic negative contribution of being female) and of specific failure modes in borderline presentations.
# - Local case analyses (true positives, true negatives, and the most unstable false positives/negatives) were used to connect quantitative fairness metrics to clinically interpretable examples. This is crucial for enabling meaningful dialogue between data scientists and medical experts.
# 
# These practices illustrate how fairness and explainability can be integrated into the model-development lifecycle, rather than treated as optional post-hoc checks.
# 

# %% [markdown]
# 
# #### 6.5 Remaining gaps and ethical recommendations
# 
# Despite these efforts, important limitations remain from an ethical standpoint. The dataset is small, historical, and geographically limited, which restricts the ability to evaluate intersectional fairness (e.g., combinations of sex, age, and phenotype) or to generalise findings to contemporary, more diverse populations. No external validation cohort was available, and the models were not co-designed or reviewed by clinicians and patient representatives, which would be essential in a real deployment context.
# 
# If a similar system were to be considered for clinical use, several additional steps would be recommended:
# 
# - Expanding the dataset to cover multiple institutions and more diverse populations, with explicit attention to underrepresented groups.
# - Incorporating fairness constraints or rebalancing strategies during training, and systematically monitoring subgroup-specific performance over time.
# - Establishing clear human-in-the-loop protocols, ensuring that clinicians remain ultimately responsible for decisions and are trained to interpret model outputs and explanations critically.
# - Creating institutional governance structures (e.g., ethics committees or AI oversight boards) to audit the system periodically, review incidents, and decide whether to adjust, retrain, or decommission the model.
# 
# Overall, this study shows that even relatively standard predictive models can exhibit clinically meaningful biases when trained on imbalanced or historically skewed data. Ethically responsible AI in healthcare therefore requires not only accurate algorithms, but also continuous data auditing, transparent explanations, and robust human oversight to prevent the reinforcement of existing inequities in diagnosis and treatment.
# 

# %% [markdown]
# ## 7. References
# 
# [1] D. Khanna, R. Sahu, V. Baths, and B. Deshpande, “Comparative Study of Classification Techniques (SVM, Logistic Regression and Neural Networks) to Predict the Prevalence of Heart Disease,” *International Journal of Machine Learning and Computing*, vol. 5, no. 5, pp. 414–419, Oct. 2015.
# 
# [2] “Machine Learning Models for Heart Disease Prediction: Performance, Robustness, and Fairness,” Internal research synthesis, Nov. 2025. (Available upon request)
# 
# [3] Artificial Intelligence and Society course material – M01: Data-Centric AI, Nov. 2025.
# 
# [4] Artificial Intelligence and Society course material – M02: Data Complexity, Nov. 2025.
# 
# [5] Artificial Intelligence and Society course material – M03: Imbalanced Data, Nov. 2025.
# 
# [6] Artificial Intelligence and Society course material – M04: Bias and Fairness, Nov. 2025.
# 
# [7] Devon, H. A., et al. (2020). *Typical and Atypical Symptoms of Acute Coronary Syndrome: Time to Retire the Terms?*  
# Journal of the American Heart Association, 9(10).  

# %% [markdown]
# 


