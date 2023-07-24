#!/usr/bin/env python3

"""
This script is used to find all ScadaBR dependencies API calls from CSV file extracted from Process Monitor

Input file is a CSV file containing all API calls, extracted from Process Monitor
Output file is all the ScadaBR dependencies API calls

Filter
(Process Name contains scadabr) ||
(Process Name is svchost.exe && Path contains scada) ||
(Process Name is services.exe && (Detail contains scada || Path contains scada)
"""

import pandas as pd

INPUT_FILE = 'input_data/All_API_Calls.CSV'
OUTPUT_FILE = 'output_data/ScadaBR_Dependencies.csv'

pd.set_option('mode.chained_assignment', None)
df = pd.read_csv(INPUT_FILE)

# Create Filters
df = df.reset_index()
submask1 = (df['Process Name'].str.contains('scadabr', case=False))
submask2 = ((df['Process Name'] == 'svchost.exe') & (df['Path'].str.contains('scada', case=False)))
submask3 = ((df['Process Name'] == 'services.exe') & ((df['Path'].str.contains('scada', case=False)) | (df['Detail'].str.contains('scada', case=False))))
mask = (submask1 | submask2 | submask3)

# Implement Filters
result = df[mask]
result = result.drop(['index'], axis=1)

# Save to file
result.to_csv(OUTPUT_FILE, index=False)