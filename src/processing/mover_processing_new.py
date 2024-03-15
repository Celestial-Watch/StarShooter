import pandas as pd
import config
import re


neg = False
add = ""

if neg:
    add = "rejected_"

# Read the CSV file
df_meta = pd.read_csv(
    config.PROCESSING_PATH + "/data/alistair/csv/" + "all_movers_metadata_in_bucket.csv"
)

df_lookup = pd.read_csv(
    config.PROCESSING_PATH + "/data/alistair/csv/" + "all_movers.csv"
)


# Count the occurrences of each Name value
name_counts = df_meta["mover_id"].value_counts()

# Filter the dataframe to include only the entries where Name appears four times
filtered_df = df_meta[df_meta["mover_id"].isin(name_counts[name_counts == 4].index)]
# Select the desired columns from the filtered dataframe
selected_df = filtered_df[["mover_id", "file_name", "totas_id", "X", "Y"]]

df_lookup.drop("file_name", axis=1, inplace=True)

# Merge the selected dataframe with the movers_pos_df based on the name equalling the mover_id column
selected_df = selected_df.merge(
    df_lookup, left_on="mover_id", right_on="mover_id", how="left"
)

# Drop any rows with empty values
selected_df.dropna(inplace=True)

# Delete Duplicate rows
selected_df = selected_df.drop_duplicates()


# Save the filtered dataframe to a CSV file

selected_df.to_csv(
    config.PROCESSING_PATH + "/data/alistair/filtered_metadata_all" + ".csv",
    index=False,
)
