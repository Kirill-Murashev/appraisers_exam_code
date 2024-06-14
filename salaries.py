import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

# Set file paths
file_path_salaries = 'salaries+.csv'
file_path_currencies = 'usd_rub.csv'
file_path_cpi = 'CPIAUCSL.csv'
file_path_oil_prices = 'oil_prices.csv'

# Load data into data frames
salaries_df = pd.read_csv(file_path_salaries)
currencies_df = pd.read_csv(file_path_currencies)
cpi_df = pd.read_csv(file_path_cpi)
oil_prices_df = pd.read_csv(file_path_oil_prices)

# Calculate the ratio of the 'official' column to the 'all' column
salaries_df['ratio'] = salaries_df['official'] / salaries_df['all']

# Multiply each value in the specified columns by the value
# in the 'ratio' column row-by-row
for col in ['appraiser', 'estimator', 'lawyer', 'auditor', 'actuary']:
    salaries_df[f'{col}_a'] = salaries_df[col] * salaries_df['ratio']

# Create a date column in 'YYYY-MM' format
salaries_df['date'] = salaries_df.apply(lambda row: f"{int(row['year'])}-{str(int(row['month'])).zfill(2)}", axis=1)

# Reorder columns to place 'date' as the first column
cols = ['date'] + [col for col in salaries_df.columns if col != 'date']
salaries_df = salaries_df[cols]

# Convert the 'date' column to datetime format
currencies_df['date'] = pd.to_datetime(currencies_df['date'], format='%m/%d/%Y')

# Filter for the relevant date range and extract year and month
filtered_currencies_df = currencies_df.loc[(currencies_df['date'] >= '2017-01-01')
                                           & (currencies_df['date'] <= '2024-05-31')]
filtered_currencies_df = filtered_currencies_df.assign(
    year=filtered_currencies_df['date'].dt.year,
    month=filtered_currencies_df['date'].dt.month
)

# Calculate the mean rate for each month
monthly_avg_rate = filtered_currencies_df.groupby(['year',
                                                   'month'])['rate'].mean().reset_index()

# Merge the monthly average rate with the salaries_df
salaries_df = salaries_df.merge(monthly_avg_rate, on=['year',
                                                      'month'], how='left')

# Rename the 'rate' column to 'monthly_avg_rate' for clarity
salaries_df.rename(columns={'rate': 'monthly_avg_rate'}, inplace=True)

# Divide the '_a' salary columns by the 'monthly_avg_rate'
# to convert salaries to USD
for col in ['appraiser_a', 'estimator_a', 'lawyer_a',
            'auditor_a', 'actuary_a', 'official']:
    salaries_df[f'{col}_usd'] = salaries_df[col] / salaries_df['monthly_avg_rate']

# Convert the 'date' column to datetime format
cpi_df['date'] = pd.to_datetime(cpi_df['date'])

# Set January 2020 as the base month
base_month = '2020-01-01'
base_index = cpi_df.loc[cpi_df['date'] == base_month, 'cpi_index'].values[0]

# Calculate the index for each month with the base month January 2020
cpi_df['index'] = cpi_df['cpi_index'] / base_index

# Extract year and month from the 'date' column for merging
cpi_df['year'] = cpi_df['date'].dt.year
cpi_df['month'] = cpi_df['date'].dt.month

# Merge the calculated index with the salaries_df
salaries_df = salaries_df.merge(cpi_df[['year', 'month', 'index']],
                                on=['year', 'month'], how='left')

# Divide each value in the specified columns by the value
# in the 'index' column row-by-row
for col in ['appraiser_a_usd', 'estimator_a_usd', 'lawyer_a_usd',
            'auditor_a_usd', 'actuary_a_usd', 'official_usd']:
    salaries_df[f'{col}_real'] = salaries_df[col] / salaries_df['index']

# Convert the 'date' column to datetime format in oil_prices_df
oil_prices_df['date'] = pd.to_datetime(oil_prices_df['date'], format='%m/%d/%Y')

# Filter for the relevant date range and extract the 'price' column
filtered_oil_prices_df = oil_prices_df.loc[(oil_prices_df['date'] >= '2017-01-01')
                                           & (oil_prices_df['date'] <= '2024-05-31'), ['date', 'price']]

# Extract year and month from the 'date' column for merging
filtered_oil_prices_df['year'] = filtered_oil_prices_df['date'].dt.year
filtered_oil_prices_df['month'] = filtered_oil_prices_df['date'].dt.month

# Merge the oil prices with the salaries_df
salaries_df = salaries_df.merge(filtered_oil_prices_df[['year', 'month', 'price']],
                                on=['year', 'month'], how='left')

# Rename the 'price' column to 'oil_price' for clarity
salaries_df.rename(columns={'price': 'oil_price'}, inplace=True)

# Calculate the salaries in barrels of oil
for col in ['appraiser_a_usd', 'estimator_a_usd', 'lawyer_a_usd',
            'auditor_a_usd', 'actuary_a_usd', 'official_usd']:
    salaries_df[f'{col.split("_")[0]}_barrels'] = salaries_df[col] / salaries_df['oil_price']

# Smooth the real salary data using the LOWESS method
for col in ['appraiser_a_usd_real', 'estimator_a_usd_real',
            'lawyer_a_usd_real', 'auditor_a_usd_real',
            'actuary_a_usd_real', 'official_usd_real']:
    salaries_df[f'{col}_smoothed'] = lowess(salaries_df[col],
                                            np.arange(len(salaries_df)),
                                            frac=0.3)[:, 1]

# Smooth the oil salary data using the LOWESS method
for col in ['appraiser_barrels', 'estimator_barrels',
            'lawyer_barrels', 'auditor_barrels',
            'actuary_barrels', 'official_barrels']:
    salaries_df[f'{col}_smoothed'] = lowess(salaries_df[col],
                                            np.arange(len(salaries_df)),
                                            frac=0.3)[:, 1]

# Display the first few rows to verify the new columns
print(salaries_df)

# Save data to CSV
salaries_df.to_csv('salaries_df.csv')

# Plot real salaries in USD
plt.figure(figsize=(21, 13))
plt.plot(salaries_df['date'], salaries_df['official_usd_real_smoothed'],
         color='black', label='Official')
plt.plot(salaries_df['date'], salaries_df['appraiser_a_usd_real_smoothed'],
         color='red', label='Appraisers')
plt.plot(salaries_df['date'], salaries_df['estimator_a_usd_real_smoothed'],
         color='blue', label='Estimators')
plt.plot(salaries_df['date'], salaries_df['lawyer_a_usd_real_smoothed'],
         color='green', label='Lawyers')
plt.plot(salaries_df['date'], salaries_df['auditor_a_usd_real_smoothed'],
         color='purple', label='Auditors')
plt.plot(salaries_df['date'], salaries_df['actuary_a_usd_real_smoothed'],
         color='orange', label='Actuaries')
plt.xlabel('Date')
plt.ylabel('Real Salaries in USD')
plt.title('Real Salaries in USD (Smoothed)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# Save the plot
plt.savefig('real_salaries_usd_smoothed.png')

# Plot salaries in barrels of oil
plt.figure(figsize=(21, 13))
plt.plot(salaries_df['date'], salaries_df['official_barrels_smoothed'],
         color='black', label='Official')
plt.plot(salaries_df['date'], salaries_df['appraiser_barrels_smoothed'],
         color='red', label='Appraisers')
plt.plot(salaries_df['date'], salaries_df['estimator_barrels_smoothed'],
         color='blue', label='Estimators')
plt.plot(salaries_df['date'], salaries_df['lawyer_barrels_smoothed'],
         color='green', label='Lawyers')
plt.plot(salaries_df['date'], salaries_df['auditor_barrels_smoothed'],
         color='purple', label='Auditors')
plt.plot(salaries_df['date'], salaries_df['actuary_barrels_smoothed'],
         color = 'orange', label='Actuaries')
plt.xlabel('Date')
plt.ylabel('Salaries in Barrels of Oil')
plt.title('Salaries in Barrels of Oil (Smoothed)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# Save the plot
plt.savefig('salaries_barrels_smoothed.png')
