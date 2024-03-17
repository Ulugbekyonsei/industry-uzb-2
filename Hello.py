####Importing
import pandas as pd
import numpy as np
import requests
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import streamlit as st
st.write(""" 
         
# Industry of Uzbekistan
        Note: This website is prepared using data from nsdp.stat.uz. The purpose of this web app is purely educational.
""") 
#####Scraping data


# URL of the Excel file on the website
excel_url = 'https://stat.uz/images/uploads/download_xlsx/nag_uzbekistan_mcd.xlsx'

# Send a GET request to fetch the Excel file
response = requests.get(excel_url, verify=False)

if response.status_code == 200:
    # Read the content of the response as an Excel file
    excel_data = response.content

    # Use io.BytesIO to create a file-like object from the content
    excel_file = io.BytesIO(excel_data)

    # Read Excel file into a Pandas DataFrame
    df = pd.read_excel(excel_file, sheet_name='Quarterly', usecols="A:AE", skiprows=5 )
    df=df.T
    # Set the second row as the column names
    # Set the second row as the column names
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    # Now you can work with the DataFrame (df) containing the Excel data
    print(df.head())  # Example: Display the first few rows of the DataFrame
else:
    print("Failed to fetch the Excel file")
s_df = df.iloc[:, 4:8]
s_df.head()
# Define the start date
start_date = '2016-01-01'

# Define the end date or the period you want to cover
end_date = '2023-06-30'  # Change this to your desired end date

# Calculate the number of quarters between the start and end dates
num_quarters = pd.date_range(start=start_date, end=end_date, freq='Q').nunique()

# Generate the date range for the quarters
quarters = pd.date_range(start=start_date, periods=num_quarters, freq='Q')

# Create a DataFrame with the quarters
time_df = pd.DataFrame({'Time': quarters})

# Assuming s_df is your existing DataFrame
s_df['Time'] = time_df['Time']
s_df['Time'] = pd.to_datetime(s_df['Time'])
# Set the DataFrame's index to be the date column
s_df.set_index('Time', inplace=True)
s_df.index =s_df.index.date
st.dataframe(s_df)
###renaming columns
s_df = s_df.rename(columns={'Mining and quarrying': 'MineQ', 'Manufacturing ': 'Man', 'Electricity, gas, steam and air conditioning supply ': 'el_gas',
                            'Water supply, sewerage, waste management and remediation ': 'wat_sup'})
plt.figure(figsize=(10, 6))  # Set the figure size
plt.stackplot(s_df.index, s_df["Man"].astype(float), s_df["MineQ"].astype(float), s_df['el_gas' ].astype(float),s_df["wat_sup" ].astype(float),   labels=['Manufacturing', 
                                                                                                  'Mining and quarrying', 'Electricity, gas, steam and air conditioning supply',
                                                                                                  'Water supply, sewerage, waste management and remediation'])
#plt.plot(s_df.index, s_df["Man"], label='Manufacturing')
#plt.plot(s_df.index, s_df["MineQ"], label='Mining and quarrying')
#plt.plot(s_df.index, s_df['el_gas' ], label='Electricity, gas, steam and air conditioning supply')
#plt.plot(s_df.index, s_df["wat_sup" ], label='Water supply, sewerage, waste management and remediation')  # Plotting the time series data
plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Value')  # Set the y-axis label
plt.title('Industry in Uzbekistan(Data source: nsdp.stat.uz)')  # Set the plot title
plt.legend()  # Show legend
plt.grid(True)  # Show grid
st.pyplot(plt)  #
s_df['MineQ_gr'] = s_df['MineQ'].pct_change(periods=4)*100
s_df['Man_gr'] = s_df['Man'].pct_change(periods=4)*100
s_df['el_gas_gr'] = s_df['el_gas'].pct_change(periods=4)*100
s_df['wat_sup_gr'] = s_df['wat_sup'].pct_change(periods=4)*100
plt.figure(figsize=(10, 6))                                                                                               
plt.plot(s_df.index, s_df["Man_gr"], label='Manufacturing')
plt.plot(s_df.index, s_df["MineQ_gr"], label='Mining and quarrying')
plt.plot(s_df.index, s_df['el_gas_gr' ], label='Electricity, gas, steam and air conditioning supply')
plt.plot(s_df.index, s_df["wat_sup_gr" ], label='Water supply, sewerage, waste management and remediation')  # Plotting the time series data
plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Value')  # Set the y-axis label
plt.title('Industry growth in Uzbekistan (Relative to preious year`s corresponding quarter)')  # Set the plot title
plt.legend()  # Show legend
plt.grid(True)  # Show grid
st.pyplot(plt)



time_index = quarters
option = st.selectbox(
    'Which indicator would you like to forecast?',
    ('Mining and quarrying', 'Manufacturing ', 'Electricity, gas, steam and air conditioning supply ',
     'Water supply, sewerage, waste management and remediation ' ))
if option == 'Mining and quarrying':
    MineQ_series=s_df['MineQ']
elif option == 'Manufacturing ':
    MineQ_series=s_df['Man']
elif option == 'Electricity, gas, steam and air conditioning supply ':
    MineQ_series=s_df['el_gas']
else:
    MineQ_series=s_df['wat_sup']


st.write('You selected:', option)
   
# Fitting an ARIMA model
# ARIMA(p, d, q) where p is the order of the autoregressive (AR) part, d is the degree of differencing, and q is the order of the moving average (MA) part
  # Replace 'Man' with the actual column name if different
MineQ_series = MineQ_series.astype(float)
# Step 3: Fit an ARIMA model
# Example: ARIMA(1, 1, 1) - You might need to adjust these parameters based on your data
model_sarima = SARIMAX(MineQ_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))  # Example seasonal_order=(P, D, Q, S)
results_sarima = model_sarima.fit()

# Step 4: Forecasting
# Specify the number of steps ahead you want to forecast
forecast_steps = 10
forecast = results_sarima.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Display the forecasted values
print(forecast_mean, forecast_conf_int)

plt.figure(figsize=(10, 7))
#plt.plot(MineQ_series, label='Original Data')
#plt.plot(pd.date_range(start=time_index[-1], periods=forecast_steps+1, freq='Q')[1:], forecast, label='Forecast', color='red')
plt.plot(MineQ_series.index, MineQ_series, label='Original Data')

# Plotting the forecast
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')

# Plotting the confidence intervals as a shaded area
plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.1, label='Confidence Interval')
plt.legend()
plt.title(f'SARIMA Forecast:{option}')
plt.xlabel('Time')
plt.ylabel('Value')
st.pyplot(plt)
st.write(f""" Forecasted values for {option} 
                 """)
data={"Forecast":forecast_mean}
f_df = pd.DataFrame(data)
f_df.index =f_df.index.date
st.dataframe(f_df)

