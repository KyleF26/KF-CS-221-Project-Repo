import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the raw data
input_file = "KF_CS221_ProjectRawData.csv"
df_raw = pd.read_csv(input_file)

# Step 2: Clean numeric data
# Drop the first column (state names) for processing
state_col = df_raw.columns[0]
state_names = df_raw[state_col]
features = df_raw.drop(columns=[state_col])

# Remove commas and convert to float
features_cleaned = features.applymap(
    lambda x: float(str(x).replace(",", "").strip()) if pd.notnull(x) else x
)

# Step 3: Normalize using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_cleaned)

# Convert back to DataFrame and reattach state column
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled.insert(0, state_col, state_names)

# Step 4: Save normalized data
output_file = "KF_CS221_ProjectNormalizedData.csv"
df_scaled.to_csv(output_file, index=False)

# Step 5: Show samples of original and normalized data
print("Sample of original data:")
print(df_raw.head(), "\n")

print("Sample of normalized data:")
print(df_scaled.head())
