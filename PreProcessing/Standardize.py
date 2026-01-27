import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('E:\Thesis\P2\completed\cleaned_dataset.csv', sep=',')

#Step-1: Transpose Data 
# Identify metadata columns (object type) and gene expression columns (numeric type)
metadata_cols = df.select_dtypes(include='object').columns.tolist()
gene_expression_cols = df.select_dtypes(include=np.number).columns.tolist()

#metadata and gene expression dataframes
metadata_df = df[metadata_cols].copy()
gene_expression_df = df[gene_expression_cols].copy()

print("Initial Gene Expression Data (first 5 rows and 5 columns, before transformations):")
print(gene_expression_df.iloc[:5, :5].to_markdown(index=False, numalign="left", stralign="left"))
print(f"Shape of gene expression data: {gene_expression_df.shape}\n")

#Step-2: Log Transform: log(x+1) 
gene_expression_log_transformed = np.log1p(gene_expression_df)

print("Log-Transformed Gene Expression Data (first 5 rows and 5 columns):")
print(gene_expression_log_transformed.iloc[:5, :5].to_markdown(index=False, numalign="left", stralign="left"))
print(f"Shape after log transformation: {gene_expression_log_transformed.shape}\n")

#Step 3: Standardize
scaler = StandardScaler()

gene_expression_standardized = pd.DataFrame(scaler.fit_transform(gene_expression_log_transformed),
                                            columns=gene_expression_log_transformed.columns,
                                            index=gene_expression_log_transformed.index)

print("Standardized Gene Expression Data (first 5 rows and 5 columns):")
print(gene_expression_standardized.iloc[:5, :5].to_markdown(index=False, numalign="left", stralign="left"))
print(f"Shape after standardization: {gene_expression_standardized.shape}\n")


#Recombine the dataset
final_cleaned_standardized_df = pd.concat([metadata_df, gene_expression_standardized], axis=1)

#gene columns with zero standard deviation
constant_gene_cols = [col for col in gene_expression_standardized.columns if gene_expression_standardized[col].std() == 0]

print(f"Number of gene columns with constant values (zero standard deviation): {len(constant_gene_cols)}")

if len(constant_gene_cols) > 0:
    # Drop
    df_cleaned_after_dropping_constant_genes = final_cleaned_standardized_df.drop(columns=constant_gene_cols)
    print(f"Dropped {len(constant_gene_cols)} gene columns.")
    print(f"New dataset shape after dropping constant gene columns: {df_cleaned_after_dropping_constant_genes.shape}\n")
    print("First 5 rows and a sample of columns from the updated dataset:")
    sample_cols = metadata_cols + [col for col in gene_expression_standardized.columns if col not in constant_gene_cols][:5]
    print(df_cleaned_after_dropping_constant_genes[sample_cols].iloc[:5, :].to_markdown(index=False, numalign="left", stralign="left"))
else:
    print("No gene columns with constant values found. No columns were dropped.")
#dropping the meta data columns that are not needed
columns_to_drop = [
     'Sample_ID',
     'ID_REF',
     'Tissue',
     'Patient_Specific_ID',
     'Cell_Type',
     'Batch',
     'Mutational_Subclass',
     'Source_Name']

columns_to_drop_existing = [col for col in columns_to_drop if col in df_cleaned_after_dropping_constant_genes.columns]
df_series_matrix_cleaned = df_cleaned_after_dropping_constant_genes.drop(columns=columns_to_drop_existing)
# print(df.columns)
# print(df.info())


print("Final Cleaned and Standardized Dataset :")
print(f"\nFinal dataset shape: {df_series_matrix_cleaned.shape}")
print("\nFinal dataset info:")
df_series_matrix_cleaned.to_csv('Final Cleaned and Standardized Dataset.csv', index=False)