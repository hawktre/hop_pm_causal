import pandas as pd
import numpy as np
import re

# Read anonymized raw data (do not use originals)
df = pd.read_csv('data/raw/cost/powdery_mildew_fungicide_record_database.csv', encoding='utf-8')
df2 = pd.read_csv('data/raw/data_2017_v2.csv', encoding='cp1252')
pesticide_price = pd.read_csv('data/raw/cost/pesticide_price.csv', encoding='utf-8')

# Quick PII guardrails
assert df['Grower'].astype(str).str.startswith('GRW_').all(), 'Expected anonymized Grower tokens in df'
assert df2['Grower'].astype(str).str.startswith('GRW_').all(), 'Expected anonymized Grower tokens in df2'

# Fix data entry errors
df['Product'] = df['Product'].str.replace('Gramoxone SL 2 ', 'Gramoxone SL 2')
df['Product'] = df['Product'].str.replace('Class Act ', 'Class Act')
df['Product'] = df['Product'].str.replace('InterLock', 'Interlock')

# Change Type column when Product = 'Preference' to 'Adjuvant' to fix data entry error
df.loc[df['Product'] == 'Preference', 'Type'] = 'Adjuvant'

# Change Type column when Product = 'Spreader 90' to 'Adjuvant' to fix data entry error
df.loc[df['Product'] == 'Spreader 90', 'Type'] = 'Adjuvant'

# Remove rows with missing values '.' in the Product column
df = df[df['Product'] != '.']

# Drop nan values from df in product column
df = df.dropna(subset=['Product'])

# Fix data entry error; mismatched June labels
df2['Month'] = df2['Month'].str.replace('June ', 'June')

# Unique Years
years = df['Year'].unique()

# Remove fields with missing values in Date column in either of the two datasets
for year in years:
    fields_to_delete = df2.loc[((df2['Date'] == '.') & (df2['Year'] == year)), 'Field ID'].tolist()
    if len(fields_to_delete) > 0:
        df2 = df2[~((df2['Field ID'].isin(fields_to_delete)) & (df2['Year'] == year))]
        df = df[~((df['Field ID'].isin(fields_to_delete)) & (df['Year'] == year))]

# Remove fields with missing values in Mildew Incidence column in either of the two datasets
for year in years:
    fields_to_delete = df2.loc[((df2['Mildew Incidence'] == '.') & (df2['Year'] == year)), 'Field ID'].tolist()
    if len(fields_to_delete) > 0:
        df2 = df2[~((df2['Field ID'].isin(fields_to_delete)) & (df2['Year'] == year))]
        df = df[~((df['Field ID'].isin(fields_to_delete)) & (df['Year'] == year))]

# Remove fields with Crystal variety in either of the two datasets
for year in years:
    fields_to_delete = df2.loc[((df2['Variety'] == 'Crystal') & (df2['Year'] == year)), 'Field ID'].tolist()
    if len(fields_to_delete) > 0:
        df2 = df2[~((df2['Field ID'].isin(fields_to_delete)) & (df2['Year'] == year))]
        df = df[~((df['Field ID'].isin(fields_to_delete)) & (df['Year'] == year))]

# MISSING VALUES
# Remove fields with missing values in Hill column in either of the two datasets
for year in years:
    fields_to_delete = df2.loc[((df2['Hill'] == '.') & (df2['Year'] == year)), 'Field ID'].tolist()
    if len(fields_to_delete) > 0:
        df2 = df2[~((df2['Field ID'].isin(fields_to_delete)) & (df2['Year'] == year))]
        df = df[~((df['Field ID'].isin(fields_to_delete)) & (df['Year'] == year))]

# Format date column
df['Spray Date'] = pd.to_datetime(df['Spray Date']).dt.strftime('%m-%d-%Y')
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%m-%d-%Y')

# Remove inconsistencies in Field IDs across datasets
for year in years:
    df_fields = df.loc[(df['Year'] == year), 'Field ID'].unique()
    df2_fields = df2.loc[(df2['Year'] == year), 'Field ID'].unique()
    fields_to_delete = np.setxor1d(df_fields, df2_fields)
    df2 = df2[~((df2['Field ID'].isin(fields_to_delete)) & (df2['Year'] == year))]
    df = df[~((df['Field ID'].isin(fields_to_delete)) & (df['Year'] == year))]

# Field IDs
field_ID = df[['Field ID', 'Year', 'Grower']]

# One hot encode the product column
dummies = pd.get_dummies(df['Product'])

# Ensure Notes and Rate can hold numeric values during conversion
# Some columns are stored with pandas string dtype and would reject float assignments.
df['Notes'] = df['Notes'].astype(object)
df['Rate'] = df['Rate'].astype(object)

# Extract numerical values from the 'Notes' column and convert gpa to oz per acre
for i in range(len(df['Notes'])):
    df.loc[df.index[i], 'Notes'] = float(re.sub(r'[^\d.]', '', str(df['Notes'].iloc[i]))) * 128

# Remove % from the 'Rate' column and multiply by oz per acre
for i in range(len(df['Rate'])):
    val = df['Rate'].iloc[i]
    if isinstance(val, str) and val.endswith('%'):
        df.loc[df.index[i], 'Rate'] = float(val.replace('%', ''))
        df.loc[df.index[i], 'Rate'] = (df['Rate'].iloc[i] / 100) * df['Notes'].iloc[i]

# Convert rates in 'lb' to 'oz' (1 lb = 16 oz) to match price units
for i in range(len(df['Rate'])):
    val = df['Rate'].iloc[i]
    if isinstance(val, str) and 'lb' in val.lower():
        rate_oz = float(re.sub(r'[^\d.]', '', val)) * 16
        df.loc[df.index[i], 'Rate'] = f"{rate_oz} oz"

# Get list of unique products
product_list = df['Product'].unique()

# Remove any product which was deleted from database from the price data
products_to_delete = np.setxor1d(product_list, pesticide_price['Product'])
pesticide_price = pesticide_price[~(pesticide_price['Product'].isin(products_to_delete))]

# Spray data and cost computation
wind_columns = (df2.columns)[-32:].tolist()
num_sprays, num_sprays_h, spray_data, spray_dummies_list, rate_list = [], [], [], [], []
field_list, year_list, grower_list, period_list = [], [], [], []
spray_rate_list, fungicide_cost_list, spray_date_list = [], [], []
spray_date_list_herb, spray_rate_list_herb, herbicide_cost_list = [], [], []
mildew_incidence_list, area_list, hill_list, w_pm_list = [], [], [], []
variety_list, sus_r6_list, sus_nonr6_list, initial_strain_list = [], [], [], []
wind_list, centroid_lat_list, centroid_long_list = [], [], []

# Early
early_year_list, early_field_list = [], []
early_spray_data, early_num_sprays, early_num_sprays_h = [], [], []
early_spray_rate_list_herb, early_spray_rate_list = [], []
early_spray_dummies_list, early_spray_date_list, early_spray_date_list_herb = [], [], []
fungicide_early_cost_list, herbicide_early_cost_list = [], []

# Late
late_year_list, late_field_list, late_grower_list = [], [], []
late_spray_data, late_num_sprays, late_num_sprays_h = [], [], []
late_spray_rate_list_herb, late_spray_rate_list = [], []
late_spray_dummies_list, late_spray_date_list, late_spray_date_list_herb = [], [], []
fungicide_late_cost_list, herbicide_late_cost_list, late_mildew_incidence_list = [], [], []

price = pesticide_price['Average R Price'].to_numpy()

for year in years:
    growers = df.loc[(df['Year'] == year), 'Grower'].unique()
    for grower in growers:
        unique_fields = df.loc[((df['Grower'] == grower) & (df['Year'] == year)), 'Field ID'].unique()
        for field in unique_fields:
            fungicide_dates = df.loc[((df['Type'] == 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field)), 'Spray Date'].unique()
            herbicide_dates = df.loc[((df['Type'] == 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field)), 'Spray Date'].unique()
            # Early season
            early_year_list.append(year)
            early_field_list.append(field)
            mask = (fungicide_dates < df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[0, 2])
            spray_dates = fungicide_dates[mask]
            early_spray_date_list.append(spray_dates)
            mask_h = (herbicide_dates < df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[0, 2])
            herb_spray_dates = herbicide_dates[mask_h]
            early_spray_date_list_herb.append(herb_spray_dates)
            mask_index, mask_index1, mask_index_herb = [], [], []
            mask_index_h, mask_index1_h, mask_index_herb_h = [], [], []
            for spray_date in spray_dates:
                mask_index_ = df.index[(df['Type'] != 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                mask_index_1 = df.index[(df['Type'] == 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                mask_index1.append(mask_index_1)
                mask_index_herb_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                for k in range(len(mask_index_)):
                    mask_index.append(mask_index_[k])
                for k in range(len(mask_index_herb_)):
                    mask_index_herb.append(mask_index_herb_[k])
            for spray_date in herb_spray_dates:
                mask_index_h_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                mask_index_1_h = df.index[(df['Type'] == 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                mask_index1_h.append(mask_index_1_h)
                for k in range(len(mask_index_h_)):
                    mask_index_h.append(mask_index_h_[k])
            spray_dummies = dummies.loc[mask_index].to_numpy()
            early_spray_dummies_list.append(spray_dummies)
            spray_dummies_1 = dummies.loc[mask_index1].to_numpy()
            spray_dummies_1_h = dummies.loc[mask_index1_h].to_numpy()
            spray_dummies_herb = dummies.loc[mask_index_herb].to_numpy()
            num_sprays_ = spray_dummies_1.sum()
            early_num_sprays.append(num_sprays_)
            num_sprays__h = spray_dummies_1_h.sum()
            early_num_sprays_h.append(num_sprays__h)
            spray_data_ = np.sum(spray_dummies, axis=0)
            early_spray_data.append(spray_data_)
            if len(mask_index) > 0 and num_sprays_ > 0:
                spray_rates = []
                rates = (df['Rate'].loc[mask_index]).to_numpy()
                for k in range(len(mask_index)):
                    rate_val = rates[k]
                    if not isinstance(rate_val, float):
                        rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                    spray_rate = rate_val * spray_dummies[k]
                    spray_rates.append(spray_rate)
                spray_rate_total = np.sum(spray_rates, axis=0)
                early_spray_rate_list.append(spray_rate_total)
                fungicide_cost = np.sum(spray_rate_total * price)
                fungicide_early_cost_list.append(fungicide_cost)
            else:
                spray_rate_total = np.zeros(dummies.shape[1], dtype=float)
                early_spray_rate_list.append(spray_rate_total)
                fungicide_early_cost_list.append(0.0)
            if len(mask_index_herb) > 0 and num_sprays__h > 0:
                spray_rates_herb = []
                rates_h = (df['Rate'].loc[mask_index_herb]).to_numpy()
                for k in range(len(mask_index_herb)):
                    rate_val = rates_h[k]
                    if not isinstance(rate_val, float):
                        rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                    spray_rate_herb = rate_val * spray_dummies_herb[k]
                    spray_rates_herb.append(spray_rate_herb)
                spray_rate_total_herb = np.sum(spray_rates_herb, axis=0)
                early_spray_rate_list_herb.append(spray_rate_total_herb)
                herbicide_cost = np.sum(spray_rate_total_herb * price)
                herbicide_early_cost_list.append(herbicide_cost)
            else:
                herbicide_early_cost_list.append(0.0)
            # Late season
            late_year_list.append(year)
            late_field_list.append(field)
            late_grower_list.append(grower)
            mask = (fungicide_dates > df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[3, 2])
            spray_dates = fungicide_dates[mask]
            late_spray_date_list.append(spray_dates)
            mask_h = (herbicide_dates > df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[3, 2])
            herb_spray_dates = herbicide_dates[mask_h]
            late_spray_date_list_herb.append(herb_spray_dates)
            mask_index, mask_index1, mask_index_herb = [], [], []
            mask_index_h, mask_index1_h, mask_index_herb_h = [], [], []
            for spray_date in spray_dates:
                mask_index_ = df.index[(df['Type'] != 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                mask_index_1 = df.index[(df['Type'] == 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                mask_index1.append(mask_index_1)
                mask_index_herb_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                for k in range(len(mask_index_)):
                    mask_index.append(mask_index_[k])
                for k in range(len(mask_index_herb_)):
                    mask_index_herb.append(mask_index_herb_[k])
            for spray_date in herb_spray_dates:
                mask_index_h_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                mask_index_1_h = df.index[(df['Type'] == 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                mask_index1_h.append(mask_index_1_h)
                for k in range(len(mask_index_h_)):
                    mask_index_h.append(mask_index_h_[k])
            spray_dummies = dummies.loc[mask_index].to_numpy()
            late_spray_dummies_list.append(spray_dummies)
            spray_dummies_1 = dummies.loc[mask_index1].to_numpy()
            spray_dummies_1_h = dummies.loc[mask_index1_h].to_numpy()
            spray_dummies_herb = dummies.loc[mask_index_herb].to_numpy()
            num_sprays_ = spray_dummies_1.sum()
            late_num_sprays.append(num_sprays_)
            num_sprays__h = spray_dummies_1_h.sum()
            late_num_sprays_h.append(num_sprays__h)
            spray_data_ = np.sum(spray_dummies, axis=0)
            late_spray_data.append(spray_data_)
            if len(mask_index) > 0 and num_sprays_ > 0:
                spray_rates = []
                rates = (df['Rate'].loc[mask_index]).to_numpy()
                for k in range(len(mask_index)):
                    rate_val = rates[k]
                    if not isinstance(rate_val, float):
                        rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                    spray_rate = rate_val * spray_dummies[k]
                    spray_rates.append(spray_rate)
                spray_rate_total = np.sum(spray_rates, axis=0)
                late_spray_rate_list.append(spray_rate_total)
                fungicide_cost = np.sum(spray_rate_total * price)
                fungicide_late_cost_list.append(fungicide_cost)
            else:
                spray_rate_total = np.zeros(dummies.shape[1], dtype=float)
                late_spray_rate_list.append(spray_rate_total)
                fungicide_late_cost_list.append(0.0)
            if len(mask_index_herb) > 0 and num_sprays__h > 0:
                spray_rates_herb = []
                rates_h = (df['Rate'].loc[mask_index_herb]).to_numpy()
                for k in range(len(mask_index_herb)):
                    rate_val = rates_h[k]
                    if not isinstance(rate_val, float):
                        rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                    spray_rate_herb = rate_val * spray_dummies_herb[k]
                    spray_rates_herb.append(spray_rate_herb)
                spray_rate_total_herb = np.sum(spray_rates_herb, axis=0)
                late_spray_rate_list_herb.append(spray_rate_total_herb)
                herbicide_cost = np.sum(spray_rate_total_herb * price)
                herbicide_late_cost_list.append(herbicide_cost)
            else:
                herbicide_late_cost_list.append(0.0)
            for t in range(4):
                field_list.append(field)
                grower_list.append(grower)
                year_list.append(year)
                mildew_incidence = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Mildew Incidence'].iloc[t]
                mildew_incidence_list.append(float(mildew_incidence))
                area = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field), 'Area_Acres'].iloc[t]
                area_list.append(float(area))
                centroid_lat = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field), 'Centroid Lat'].iloc[t]
                centroid_lat_list.append(float(centroid_lat))
                centroid_long = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field), 'Centroid Long'].iloc[t]
                centroid_long_list.append(float(centroid_long))
                hill = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Hill'].iloc[t]
                hill_list.append(int(hill))
                w_pm = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'w/PM'].iloc[t]
                w_pm_list.append(w_pm)
                variety = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Variety'].iloc[t]
                variety_list.append(variety)
                sus_r6 = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Susceptibility to R6 Strains'].iloc[t]
                sus_r6_list.append(int(sus_r6))
                sus_nonr6 = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Susceptibility to non-R6 Strains'].iloc[t]
                sus_nonr6_list.append(int(sus_nonr6))
                initial_strain = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field) , 'Initial Strain'].iloc[t]
                initial_strain_list.append(initial_strain)
                wind_pre_list = []
                for i in wind_columns:
                    wind = df2.loc[(df2['Year'] == year) & (df2['Grower'] == grower) & (df2['Field ID'] == field), i].iloc[t]
                    wind_pre_list.append(float(wind))
                wind_list.append(wind_pre_list)
                if t == 0:
                    period_list.append('April')
                    spray_dummies = np.zeros(dummies.shape[1], dtype=int)
                    spray_dummies_list.append(spray_dummies)
                    num_sprays.append(0)
                    num_sprays_h.append(0)
                    spray_data.append(np.zeros(dummies.shape[1], dtype=int))
                    spray_rate_list.append(np.zeros(dummies.shape[1], dtype=float))
                    fungicide_cost_list.append(0.0)
                    herbicide_cost_list.append(0.0)
                else:
                    mask = ((fungicide_dates >= df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[t-1, 2]) & (fungicide_dates < df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[t, 2]))
                    spray_dates = fungicide_dates[mask]
                    spray_date_list.append(spray_dates)
                    mask_h = ((herbicide_dates >= df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[t-1, 2]) & (herbicide_dates < df2[(df2['Field ID'] == field) & (df2['Year'] == year)].iloc[t, 2]))
                    herb_spray_dates = herbicide_dates[mask_h]
                    spray_date_list_herb.append(herb_spray_dates)
                    mask_index, mask_index1, mask_index_herb = [], [], []
                    mask_index_h, mask_index1_h, mask_index_herb_h = [], [], []
                    for spray_date in spray_dates:
                        mask_index_ = df.index[(df['Type'] != 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                        mask_index_1 = df.index[(df['Type'] == 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                        mask_index1.append(mask_index_1)
                        mask_index_herb_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                        for k in range(len(mask_index_)):
                            mask_index.append(mask_index_[k])
                        for k in range(len(mask_index_herb_)):
                            mask_index_herb.append(mask_index_herb_[k])
                    for spray_date in herb_spray_dates:
                        mask_index_h_ = df.index[(df['Type'] != 'Fungicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)]
                        mask_index_1_h = df.index[(df['Type'] == 'Herbicide') & (df['Year'] == year) & (df['Grower'] == grower) & (df['Field ID'] == field) & (df['Spray Date'] == spray_date)][0]
                        mask_index1_h.append(mask_index_1_h)
                        for k in range(len(mask_index_h_)):
                            mask_index_h.append(mask_index_h_[k])
                    spray_dummies = dummies.loc[mask_index].to_numpy()
                    spray_dummies_list.append(spray_dummies)
                    spray_dummies_1 = dummies.loc[mask_index1].to_numpy()
                    spray_dummies_1_h = dummies.loc[mask_index1_h].to_numpy()
                    spray_dummies_herb = dummies.loc[mask_index_herb].to_numpy()
                    num_sprays_ = spray_dummies_1.sum()
                    num_sprays.append(num_sprays_)
                    num_sprays__h = spray_dummies_1_h.sum()
                    num_sprays_h.append(num_sprays__h)
                    spray_data_ = np.sum(spray_dummies, axis=0)
                    spray_data.append(spray_data_)
                    if len(mask_index) > 0 and num_sprays_ > 0:
                        spray_rates = []
                        rates = (df['Rate'].loc[mask_index]).to_numpy()
                        for k in range(len(mask_index)):
                            rate_val = rates[k]
                            if not isinstance(rate_val, float):
                                rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                            spray_rate = rate_val * spray_dummies[k]
                            spray_rates.append(spray_rate)
                        spray_rate_total = np.sum(spray_rates, axis=0)
                        spray_rate_list.append(spray_rate_total)
                        fungicide_cost = np.sum(spray_rate_total * price)
                        fungicide_cost_list.append(fungicide_cost)
                    else:
                        spray_rate_total = np.zeros(dummies.shape[1], dtype=float)
                        spray_rate_list.append(spray_rate_total)
                        fungicide_cost_list.append(0.0)
                    if len(mask_index_herb) > 0 and num_sprays__h > 0:
                        spray_rates_herb = []
                        rates_h = (df['Rate'].loc[mask_index_herb]).to_numpy()
                        for k in range(len(mask_index_herb)):
                            rate_val = rates_h[k]
                            if not isinstance(rate_val, float):
                                rate_val = float(re.sub(r'[^\d.]', '', str(rate_val)))
                            spray_rate_herb = rate_val * spray_dummies_herb[k]
                            spray_rates_herb.append(spray_rate_herb)
                        spray_rate_total_herb = np.sum(spray_rates_herb, axis=0)
                        spray_rate_list_herb.append(spray_rate_total_herb)
                        herbicide_cost = np.sum(spray_rate_total_herb * price)
                        herbicide_cost_list.append(herbicide_cost)
                    else:
                        herbicide_cost_list.append(0.0)
                if t == 1:
                    period_list.append('May')
                elif t == 2:
                    period_list.append('June')
                elif t == 3:
                    period_list.append('July')

# Convert lists to arrays
spray_data = np.array(spray_data)
spray_rate_array = np.array(spray_rate_list)

# Create dictionary of field IDs, year, grower, sprays, and cost
cost_data = {
    'Field ID': field_list,
    'Year': year_list,
    'Month': period_list,
    'Centroid Lat': centroid_lat_list,
    'Centroid Long': centroid_long_list,
    'Grower': grower_list,
    'Sprays': num_sprays,
    'Herbicide Sprays': num_sprays_h,
    'Fungicide Cost': fungicide_cost_list,
    'Herbicide Cost': herbicide_cost_list,
    'Area_Acres': area_list,
    'Mildew Incidence': mildew_incidence_list,
    'Hill': hill_list,
    'w/PM': w_pm_list,
    'Variety': variety_list,
    'Susceptibility to R6 Strains': sus_r6_list,
    'Susceptibility to non-R6 Strains': sus_nonr6_list,
    'Initial Strain': initial_strain_list
}
cost_data = pd.DataFrame(cost_data)

# Create dictionary for early season cost
early_cost_data = {
    'Field ID': early_field_list,
    'Year': early_year_list,
    'Sprays': early_num_sprays,
    'Herbicide Sprays': early_num_sprays_h,
    'Fungicide Cost': fungicide_early_cost_list,
    'Herbicide Cost': herbicide_early_cost_list
}
early_cost_data = pd.DataFrame(early_cost_data)

# Create dictionary for late season cost
late_cost_data = {
    'Field ID': late_field_list,
    'Year': late_year_list,
    'Sprays': late_num_sprays,
    'Herbicide Sprays': late_num_sprays_h,
    'Fungicide Cost': fungicide_late_cost_list,
    'Herbicide Cost': herbicide_late_cost_list
}
late_cost_data = pd.DataFrame(late_cost_data)

# Create wind data frame
wind_dict = {}
keys = wind_columns
values = (np.array(wind_list).T).tolist()
for i in range(len(keys)):
    wind_dict[keys[i]] = values[i]
wind_dict = pd.DataFrame(wind_dict)
for i in wind_columns:
    cost_data[i] = wind_dict[i]

# Convert area from acres to hectares
cost_data['Area_Acres'] = cost_data['Area_Acres'].astype(float)
cost_data['Area_Acres'] = cost_data['Area_Acres'] * 0.404686
cost_data.rename(columns={'Area_Acres': 'Area_Hectares'}, inplace=True)

# Include application costs
fung_application_cost = 16.0 * 2.471053814671653
herb_application_cost = 9.63 * 2.471053814671653

# PPI (Jan 2022 base) for 2014-2020
ppi = [89.63222295, 78.91586143, 71.31375327, 73.47443209, 73.05062119, 73.34955924, 71.70729197, 89.92332807, 107.9811162]
ppi_2020 = ppi[6]
ppi_2022 = ppi[8]

# Adjust application cost for inflation
fung_application_cost = fung_application_cost / ppi_2022 * 100
herb_application_cost = herb_application_cost / ppi_2020 * 100

# Convert cost from dollars per acre to dollars per hectare
cost_data['Fungicide Cost'] = cost_data['Fungicide Cost'] * 2.471053814671653
cost_data['Herbicide Cost'] = cost_data['Herbicide Cost'] * 2.471053814671653

early_cost_data['Fungicide Cost'] = early_cost_data['Fungicide Cost'] * 2.471053814671653
early_cost_data['Herbicide Cost'] = early_cost_data['Herbicide Cost'] * 2.471053814671653

late_cost_data['Fungicide Cost'] = late_cost_data['Fungicide Cost'] * 2.471053814671653
late_cost_data['Herbicide Cost'] = late_cost_data['Herbicide Cost'] * 2.471053814671653

# Include application costs in herbicide cost (no application cost included for fungicides in mid-season)
cost_data['Herbicide Cost'] = cost_data['Herbicide Cost'].values + (herb_application_cost * cost_data['Herbicide Sprays']).values

# Include application costs for fungicides and herbicides for early and late season only
late_cost_data['Herbicide Cost'] = late_cost_data['Herbicide Cost'].values + (herb_application_cost * late_cost_data['Herbicide Sprays']).values
early_cost_data['Herbicide Cost'] = early_cost_data['Herbicide Cost'].values + (herb_application_cost * early_cost_data['Herbicide Sprays']).values
early_cost_data['Fungicide Cost'] = early_cost_data['Fungicide Cost'].values + (fung_application_cost * early_cost_data['Sprays']).values

# Assign data types
cost_data.iloc[:, -32:] = cost_data.iloc[:, -32:].astype(float)
cost_data['Month'] = cost_data['Month'].astype('string')
cost_data['Grower'] = cost_data['Grower'].astype('string')
cost_data['Variety'] = cost_data['Variety'].astype('string')
cost_data['Mildew Incidence'] = cost_data['Mildew Incidence'].astype(float)
cost_data['w/PM'] = cost_data['w/PM'].astype(float)
cost_data['Sprays'] = cost_data['Sprays'].astype(float)
cost_data['Hill'] = cost_data['Hill'].astype(float)
cost_data['Centroid Lat'] = cost_data['Centroid Lat'].astype(float)
cost_data['Centroid Long'] = cost_data['Centroid Long'].astype(float)

early_cost_data['Sprays'] = early_cost_data['Sprays'].astype(float)
late_cost_data['Sprays'] = late_cost_data['Sprays'].astype(float)


# DATA CLEANING
# Map initial strain terminology (R6 to V6, non-R6 to non-V6)
cost_data['Initial Strain'] = cost_data['Initial Strain'].replace({'R6': 'V6', 'non-R6': 'non-V6'})

# Derive cultivar category from susceptibility to non-V6 strains
cost_data['Cultivar'] = np.where(
    (cost_data['Susceptibility to non-R6 Strains'] > 0),
    'non-R6', 'R6'
)
# Confirm initial strain using cultivar
cost_data.loc[cost_data['Cultivar'] == 'R6', 'Initial Strain'] = 'V6'
cost_data.loc[(cost_data['Cultivar'] == 'non-R6') & (cost_data['Initial Strain'] == '.'), 'Initial Strain'] = 'unknown'

# Rename to V6 terminology and binarize susceptibility columns
cost_data.rename(columns={
    'Susceptibility to R6 Strains': 'Susceptibility to V6 Strains',
    'Susceptibility to non-R6 Strains': 'Susceptibility to non-V6 Strains'
}, inplace=True)
cost_data['Susceptibility to V6 Strains'] = (cost_data['Susceptibility to V6 Strains'] > 0).astype(int)
cost_data['Susceptibility to non-V6 Strains'] = (cost_data['Susceptibility to non-V6 Strains'] > 0).astype(int)

# Encode as binary: V6=1, non-V6 or unknown=0
cost_data['Initial Strain'] = np.where(cost_data['Initial Strain'] == "V6", 1, 0)
cost_data['Cultivar'] = np.where(cost_data['Cultivar'] == "R6", 1, 0)

print('Preprocessing (anonymized) complete.')

# Export anonymized outputs
cost_data.to_csv('data/processed/cost_data.csv', index=False)
early_cost_data.to_csv('data/processed/early_cost_data.csv', index=False)
late_cost_data.to_csv('data/processed/late_cost_data.csv', index=False)
print('Wrote anonymized processed datasets.')
