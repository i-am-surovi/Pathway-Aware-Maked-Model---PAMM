import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the GSE68086_series_matrix.csv file
df_series_matrix = pd.read_csv(r'E:\Thesis\P2\completed\Updated GSE68086_series_matrix.csv', encoding='latin1')

#Identify and drop unnecessary columns
columns_to_drop = [
     '!Sample_status',
     '!Sample_submission_date',
     '!Sample_last_update_date',
     '!Sample_type',
     '!Sample_channel_count',
     '!Sample_organism_ch1',
     '!Sample_molecule_ch1',
     '!Sample_extract_protocol_ch1',
     '!Sample_extract_protocol_ch1.1',
     '!Sample_taxid_ch1',
     '!Sample_description',
     '!Sample_data_processing',
     '!Sample_data_processing.1',
     '!Sample_data_processing.2',
     '!Sample_data_processing.3',
     '!Sample_data_processing.4',
     '!Sample_data_processing.5',
     '!Sample_platform_id',
     '!Sample_contact_name',
     '!Sample_contact_email',
     '!Sample_contact_laboratory',
     '!Sample_contact_department',
     '!Sample_contact_institute',
     '!Sample_contact_address',
     '!Sample_contact_city',
     '!Sample_contact_zip/postal_code',
     '!Sample_contact_country',
     '!Sample_data_row_count',
     '!Sample_instrument_model',
     '!Sample_library_selection',
     '!Sample_library_source',
     '!Sample_library_strategy',
     '!Sample_relation',
     '!Sample_relation.1',
     '!Sample_supplementary_file_1',
     '!series_matrix_table_begin', # These were NaN and are file format markers
     '!series_matrix_table_end'    # These were NaN and are file format markers
 ]

# # Ensure only columns that exist in the DataFrame are dropped
columns_to_drop_existing = [col for col in columns_to_drop if col in df_series_matrix.columns]
df_series_matrix_cleaned = df_series_matrix.drop(columns=columns_to_drop_existing)
# print(df_series_matrix_cleaned.info())

#Extract and rename information from !Sample_characteristics_ch1.X columns
# # Define a function to extract the value after ': '
def extract_value(text):
    if isinstance(text, str) and ': ' in text:
        return text.split(': ', 1)[1].strip().rstrip('"')
    return None

# # Rename !Sample_geo_accession upfront for clarity and merging later
df_series_matrix_cleaned = df_series_matrix_cleaned.rename(columns={'!Sample_geo_accession': 'Sample_ID'})
df_series_matrix_cleaned = df_series_matrix_cleaned.rename(columns={'!Sample_source_name_ch1': 'Source_Name'})
df_series_matrix_cleaned = df_series_matrix_cleaned.rename(columns={'"ID_REF"': 'ID_REF'})

 # Apply extraction to relevant characteristics columns
df_series_matrix_cleaned['Tissue'] = df_series_matrix_cleaned['!Sample_characteristics_ch1'].apply(extract_value)
df_series_matrix_cleaned['Cell_Type'] = df_series_matrix_cleaned['!Sample_characteristics_ch1.1'].apply(extract_value)
df_series_matrix_cleaned['Patient_Specific_ID'] = df_series_matrix_cleaned['!Sample_characteristics_ch1.2'].apply(extract_value)
df_series_matrix_cleaned['Cancer_Type'] = df_series_matrix_cleaned['!Sample_characteristics_ch1.3'].apply(extract_value)
df_series_matrix_cleaned['Batch'] = df_series_matrix_cleaned['!Sample_characteristics_ch1.4'].apply(extract_value)
df_series_matrix_cleaned['Mutational_Subclass'] = df_series_matrix_cleaned['!Sample_characteristics_ch1.5'].apply(extract_value)

# Drop the original !Sample_characteristics_ch1.X columns after extraction
columns_to_drop_characteristics = [
     '!Sample_characteristics_ch1',
     '!Sample_characteristics_ch1.1',
     '!Sample_characteristics_ch1.2',
     '!Sample_characteristics_ch1.3',
     '!Sample_characteristics_ch1.4',
     '!Sample_characteristics_ch1.5'
 ]
columns_to_drop_characteristics_existing = [col for col in columns_to_drop_characteristics if col in df_series_matrix_cleaned.columns]
df_series_matrix_cleaned = df_series_matrix_cleaned.drop(columns=columns_to_drop_characteristics_existing)
# print(df_series_matrix_cleaned.info())


# Drop rows with specific cancer types
cancer_types_to_exclude = ['Hepatobiliary', 'CRC', 'Pancreas']
df_series_matrix_final = df_series_matrix_cleaned[~df_series_matrix_cleaned['Cancer_Type'].isin(cancer_types_to_exclude)].copy()
print(f"\nNumber of rows in metadata after cleaning and filtering: {len(df_series_matrix_final)}\n")


# # # Display the first few rows and info of the cleaned DataFrame to confirm
# #print("--- Cleaned GSE68086_series_matrix.csv Info ---")
# # print(df_series_matrix_final.info())
# # print("\n--- Cleaned GSE68086_series_matrix.csv Head ---")
# # print(df_series_matrix_final.head())

# # print(f"\nOriginal number of rows: {len(df_series_matrix)}")
# # print(f"Number of rows after filtering specific cancer types: {len(df_series_matrix_final)}")
# # # show only the cancer types present in the final DataFrame
# # print(df_series_matrix_final['Cancer_Type'].unique())



# Load GSE68086_TEP_data_matrix.csv
df_tep_data_matrix = pd.read_csv('E:\Thesis\P2\completed\GSE68086_TEP_data_matrix.csv')
# print(df_tep_data_matrix.info())

# --- Debugging from previous run showed first column is 'Unnamed: 0' ---
# Rename the first column to 'ProbeID' as it contains gene identifiers
df_tep_data_matrix = df_tep_data_matrix.rename(columns={df_tep_data_matrix.columns[0]: 'ProbeID'})
# print(df_tep_data_matrix.columns)

# Clean column names of df_tep_data_matrix (these are the sample source names)
# Remove quotes and any leading/trailing whitespace
cleaned_tep_columns = {col: col.replace('"', '').strip() for col in df_tep_data_matrix.columns if col != 'ProbeID'}
df_tep_data_matrix = df_tep_data_matrix.rename(columns=cleaned_tep_columns)
# print(len(df_tep_data_matrix.columns))

# Transpose GSE68086_TEP_data_matrix.csv
# Set 'ProbeID' as index, then transpose. The new index will be the sample names (e.g., '3-Breast-Her2-ampl')
df_tep_data_transposed = df_tep_data_matrix.set_index('ProbeID').T
df_tep_data_transposed.index.name = 'Source_Name' # Name the index for merging
# print(df_tep_data_transposed.columns)

# Clean the 'Sample_source_name_ch1' column in the metadata to match the transposed gene expression index
df_series_matrix_final['Source_Name'] = df_series_matrix_final['Source_Name'].str.replace('"', '').str.strip()

# Get the list of `Sample_source_name_ch1` values that are present in the *cleaned* metadata
sample_source_names_to_keep = df_series_matrix_final['Source_Name'].unique()
# print(len(sample_source_names_to_keep))

# Filter Transposed Gene Expression Data
# Select only those rows (samples) in the transposed gene expression data
# whose 'Sample_Source_Name' is in our filtered metadata's 'Sample_source_name_ch1' list.
df_tep_data_filtered = df_tep_data_transposed.reindex(
      df_tep_data_transposed.index.intersection(sample_source_names_to_keep)).copy()
# print(df_tep_data_filtered.info)

# Merge DataFrames
# Merge based on 'Sample_source_name_ch1' from metadata and the index ('Sample_Source_Name') of gene expression
merged_data = pd.merge(
      df_series_matrix_final,
      df_tep_data_filtered,
      left_on='Source_Name', # Column in metadata
      right_index=True,                  # Index of the gene expression data
      how='inner'                        # Use inner merge to keep only matching samples
  )
# print(merged_data.info())

#  Final clean on selected characteristics columns in the merged data
merged_data['Tissue'] = merged_data['Tissue'].str.strip()
merged_data['Cell_Type'] = merged_data['Cell_Type'].str.strip()
merged_data['Patient_Specific_ID'] = merged_data['Patient_Specific_ID'].str.strip()
merged_data['Cancer_Type'] = merged_data['Cancer_Type'].str.strip()
merged_data['Batch'] = merged_data['Batch'].str.strip()
merged_data['Mutational_Subclass'] = merged_data['Mutational_Subclass'].str.strip()



# # Display info and head of the final merged DataFrame
# print("\n--- Final Merged Data Info ---")
print(merged_data.info())
# print("\n--- Final Merged Data Head ---")
# print(merged_data.head())
# print(len(merged_data))

# # print(f"\nNumber of samples in final merged dataset: {len(merged_data)}")

# merged_data.to_csv('cleaned_dataset.csv', index=False)

df_clean_matrix = pd.read_csv(r'E:\Thesis\P2\completed\cleaned_dataset.csv', encoding='latin1')
print(df_clean_matrix.info())
