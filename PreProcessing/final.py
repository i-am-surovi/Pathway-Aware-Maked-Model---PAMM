import pandas as pd
from scipy.stats import f_oneway # use to perform one-way ANOVA
import statsmodels.stats.multitest as smm
import numpy as np
# NEW IMPORTS FOR CROSS-VALIDATED METHODS
from sklearn.linear_model import LogisticRegressionCV # For Lasso with CV
from sklearn.feature_selection import RFECV # For Cross-Validated RFE
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold # For CV in RFECV and stability metric

# --- Start of Original Code (Data Loading and Initial Filtering) ---

# NOTE: The path for gene_data must be valid for the code to run.
gene_data = pd.read_csv(r'E:\Thesis\Final Cleaned and Standardized Dataset.csv', encoding='latin1')

print("Loaded data shape:", gene_data.shape)
print("Columns:", gene_data.columns[:5])
print("First 5 rows:\n", gene_data.head())

# 1. Unique value of target column
target_column = 'Cancer_Type'
gene_columns = [col for col in gene_data.columns if col != target_column]
Cancer_Types = gene_data[target_column].unique()
# print(Cancer_Types)

# --- ANOVA analysis ---
p_values = {}
for gene in gene_columns:
    groups = [gene_data[gene][gene_data[target_column] == ct] for ct in Cancer_Types]
    f_statistic, p_value = f_oneway(*groups)
    p_values[gene] = p_value

p_values_series = pd.Series(p_values)

# --- Multiple Hypothesis Correction and Gene Selection (FDR) ---
reject, p_values_corrected, _, _ = smm.multipletests(p_values_series, method='fdr_bh')
anova_results = pd.DataFrame({
    'p_value_uncorrected': p_values_series,
    'p_value_corrected': p_values_corrected,
    'reject_null': reject
})
alpha = 0.05
significant_genes = anova_results[anova_results['p_value_corrected'] < alpha]
final_gene_list = significant_genes.index.tolist()
reduced_data = gene_data[final_gene_list + [target_column]]

print(f"Total number of genes: {len(gene_columns)}")
print(f"Number of significant genes after FDR correction: {len(significant_genes)}")
print("\nShape of the original data:", gene_data.shape)
print("Shape of the reduced data:", reduced_data.shape)

print("---"*25)

mean_expression = gene_data.groupby('Cancer_Type')[final_gene_list].mean()

print("Mean expression per cancer type:\n", mean_expression.head())
print("\nShape of mean expression data:", mean_expression.shape)

# 2. Log2 Fold Change (log2FC) for all pairwise comparisons
log2_fc_threshold = 1 # a 2-fold change

# A list to store genes that pass the log2FC filter
genes_with_significant_fc = []

# Iterate through each significant gene from the ANOVA results
for gene in final_gene_list:
    
    # mean expression values for this gene across all cancer types
    expression_values = mean_expression[gene]
    
    # Check all pairwise combinations of cancer types
    has_high_fc = False
    for i in range(len(Cancer_Types)):
        for j in range(i + 1, len(Cancer_Types)):
            ct1 = Cancer_Types[i]
            ct2 = Cancer_Types[j]
            
            # a pseudocount (e.g., 1) to avoid log(0) errors
            val1 = expression_values[ct1] + 1
            val2 = expression_values[ct2] + 1
            
            # Calculate the log2 fold change
            log2_fc = np.log2(val1 / val2)
            
            # Check if the absolute log2FC is greater than the threshold
            if abs(log2_fc) >= log2_fc_threshold:
                has_high_fc = True
                break
        if has_high_fc:
            break
            
    # If the gene has a high fold change in at least one comparison, keep it
    if has_high_fc:
        genes_with_significant_fc.append(gene)

# --- Step 3: Filter the data using the new, smaller gene list ---
final_final_gene_list = genes_with_significant_fc
reduced_data_with_fc = gene_data[final_final_gene_list + ['Cancer_Type']]

print(f"\nNumber of genes after ANOVA: {len(final_gene_list)}")
print(f"Number of genes after adding a Fold Change filter: {len(final_final_gene_list)}")
print("Shape of the final reduced data:", reduced_data_with_fc.shape)

print("---"*25)
# now, 194 rows and 5177 (gene columns + 'Cancer_Type' column)
# End of original data preparation. Start of supervisor-requested changes.

# --- Prepare the data for the model ---
X = reduced_data_with_fc.drop('Cancer_Type', axis=1)
y = reduced_data_with_fc['Cancer_Type']

# Convert categorical target 'y' to numerical labels for the model
# Label encoding for OneVsRestClassifier
y_encoded = y.astype('category').cat.codes

# --- Standardize the features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# --- 5. Lasso Logistic Regression with Cross-Validation (LassoCV) ---
# Goal: Find the optimal regularization parameter (C or alpha) and select genes.
# C (in LogisticRegression) is the inverse of the regularization strength (alpha or lambda).
print("\n--- 5. Lasso Logistic Regression with Cross-Validation ---")

# Define the LogisticRegressionCV with L1 penalty (Lasso)
# OneVsRestClassifier is needed to apply the binary classifier across multiple classes.
# The CV process will run for each binary classifier.
lasso_cv_model = OneVsRestClassifier(
    LogisticRegressionCV(
        Cs=10, # Number of C values to test (inversely related to lambda)
        penalty='l1', 
        solver='liblinear',
        cv=StratifiedKFold(5), # Use 5-fold Stratified CV for stable results
        scoring='f1_macro', # Use F1-macro for multi-class imbalanced data
        random_state=42, 
        max_iter=5000,
        n_jobs=-1
    )
)

lasso_cv_model.fit(X_scaled, y_encoded)

# Extract coefficients and selected genes
feature_coefficients = np.concatenate([model.coef_ for model in lasso_cv_model.estimators_], axis=0)
nonzero_indices = np.where(np.any(feature_coefficients != 0, axis=0))[0]
final_gene_list_lasso = X.columns[nonzero_indices].tolist()

# Get the best C value used by the model for the stability metric mention
best_C_values = [model.C_[0] for model in lasso_cv_model.estimators_]
stability_metric = np.mean(best_C_values)

print(f"Optimal regularization C values (one per class): {best_C_values}")
print(f"Average optimal C (Stability Metric): {stability_metric:.4f}")
print(f"Number of genes selected after LassoCV: {len(final_gene_list_lasso)}")

# --- Create the DataFrame for the next step ---
final_reduced_data = reduced_data_with_fc[final_gene_list_lasso + ['Cancer_Type']]

print("Shape of the data after LassoCV:", final_reduced_data.shape)

print("---"*25)

# now, 194 rows, ~341 (gene columns + 'Cancer_Type' column) in the original example.

# --- 6. Recursive Feature Elimination with Cross-Validation (RFECV) ---
# Goal: Automatically find the optimal number of features.
print("\n--- 6. Recursive Feature Elimination with Cross-Validation (RFECV) ---")

X_rfe = final_reduced_data.drop('Cancer_Type', axis=1)
y_rfe = final_reduced_data['Cancer_Type']

# Standardize the features (necessary for distance-based estimators like RF's internal process)
scaler_rfe = StandardScaler()
X_rfe_scaled = scaler_rfe.fit_transform(X_rfe)
X_rfe_scaled = pd.DataFrame(X_rfe_scaled, columns=X_rfe.columns)

# Initialize the estimator (Random Forest) and RFECV
estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Use StratifiedKFold for cross-validation within RFE
cv_rfe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfe_selector = RFECV(
    estimator, 
    step=1, # Number of features to remove at each iteration
    cv=cv_rfe, 
    scoring='accuracy', # Scoring metric for RFE optimization
    min_features_to_select=5, # Minimum number of features to keep
    n_jobs=-1
)

# Run RFECV to select the optimal number of genes
rfe_selector.fit(X_rfe_scaled, y_rfe)

# Get the names of the selected genes
final_gene_list_rfe = X_rfe.columns[rfe_selector.support_].tolist()

# --- Step 4: Create the final DataFrame ---
final_final_data = final_reduced_data[final_gene_list_rfe + ['Cancer_Type']]

print(f"Number of genes before RFECV: {len(X_rfe.columns)}")
print(f"Optimal number of genes determined by RFECV: {rfe_selector.n_features_}")
# CORRECTED LINE BELOW:
print(f"Cross-Validation Score for the optimal number of features: {rfe_selector.cv_results_['mean_test_score'][rfe_selector.n_features_ - 1]:.4f}")
print("Shape of the final reduced data:", final_final_data.shape)

# Save the final dataset
final_final_data.to_csv('Final_CV_Selected.csv', index=False)