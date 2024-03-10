import pandas as pd
import config
import re


neg = True
add = ""

if neg:
    add = "rejected_"

# Read the CSV file
df = pd.read_csv(config.PROCESSING_PATH + '/data/alistair/new/'+add+'movers_metadata_new.csv')
movers_pos_df = pd.read_csv(config.PROCESSING_PATH + '/data/alistair/new/'+add+'movers_small_images.csv')

# Count the occurrences of each Name value
name_counts = df['Name'].value_counts()

# Filter the dataframe to include only the entries where Name appears four times
filtered_df = df[df['Name'].isin(name_counts[name_counts == 4].index)]
# Select the desired columns from the filtered dataframe
selected_df = filtered_df[['Link', 'X', 'Y', 'Name']]

# Apply regex to extract the id from the filename
selected_df['image_id'] = selected_df['Link'].apply(lambda x: re.search(r'(\d+)', x).group(1))

# Drop the 'filename' column
selected_df.drop('Link', axis=1, inplace=True)

# Merge the selected dataframe with the movers_pos_df based on the name equalling the mover_id column
selected_df = selected_df.merge(movers_pos_df, left_on='Name', right_on='mover_id', how='left')

# Drop any rows with empty values
selected_df.dropna(inplace=True)

selected_df['Seq_no'] = selected_df['centred_image_id'].apply(lambda x: int(re.search(r'\-(\d+)\.png$', x).group(1)))

selected_df['Occurrence'] = selected_df.groupby('centred_image_id').cumcount()
selected_df = selected_df[selected_df['Occurrence'] == (selected_df['Seq_no'] - 1) ].drop('Seq_no', axis = 1).drop('Occurrence', axis = 1)



# Save the filtered dataframe to a CSV file
name = "pos"
if neg:
    name = "neg"
selected_df.to_csv(config.PROCESSING_PATH + '/data/alistair/filtered_metadata_'+name+'.csv', index=False)