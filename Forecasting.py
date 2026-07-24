import pandas as pd

df = pd.read_csv('Dataset.csv')

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Drop rows where date_added couldn't be parsed (missing/malformed dates)
print("Rows before dropping missing dates:", len(df))
df = df.dropna(subset=['date_added'])
print("Rows after dropping missing dates:", len(df))

# Extract useful time components
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['year_month'] = df['date_added'].dt.to_period('M')

print(df[['title', 'date_added', 'year_added', 'month_added']].head(5).to_string())
print()
print("Date range:", df['date_added'].min(), "to", df['date_added'].max())
import matplotlib.pyplot as plt

# Step 2: Analyze historical release patterns
# Count how many titles were added in each year-month
monthly_releases = df.groupby('year_month').size()
monthly_releases.index = monthly_releases.index.to_timestamp()  # convert Period back to datetime for plotting

print("Monthly release counts (first 5):")
print(monthly_releases.head())
print("\nMonthly release counts (last 5):")
print(monthly_releases.tail())
print("\nAverage titles added per month:", monthly_releases.mean().round(2))

# Yearly totals -- easier to see the big-picture growth trend
yearly_releases = df.groupby('year_added').size()
print("\nYearly totals:")
print(yearly_releases)

# Plot the monthly trend
plt.figure(figsize=(14, 6))
plt.plot(monthly_releases.index, monthly_releases.values)
plt.xlabel('Date')
plt.ylabel('Titles Added')
plt.title('Netflix Monthly Content Additions Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ts = monthly_releases[monthly_releases.index <= '2021-08-01']

print("Time series length (months):", len(ts))
print(ts.tail(10))

# Split into train/test: hold back the last 6 months to evaluate forecast accuracy later
train = ts[:-6]
test = ts[-6:]

print("\nTraining months:", len(train))
print("Test months:", len(test))

model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=12
)
fitted_model = model.fit()

print("\nModel fitted successfully.")
print(fitted_model.summary())
test_forecast = fitted_model.forecast(6)

print("Forecast vs Actual (test period):")
comparison = pd.DataFrame({
    'Actual': test.values,
    'Forecast': test_forecast.values.round(1)
}, index=test.index)
print(comparison)

full_model = ExponentialSmoothing(
    ts, trend='add', seasonal='add', seasonal_periods=12
).fit()

future_forecast = full_model.forecast(6)
print("\nForecast for next 6 months (beyond available data):")
print(future_forecast.round(1))

# Fix: build proper future dates for the forecast, since .forecast() returns
# a plain integer index by default, not real dates.
last_date = ts.index[-1]
future_dates = pd.date_range(start=last_date, periods=7, freq='MS')[1:]  # skip the last known date itself
future_forecast.index = future_dates

print("\nForecast for next 6 months (with proper dates):")
print(future_forecast.round(1))

# Plot actual history + future forecast together
plt.figure(figsize=(14, 6))
plt.plot(ts.index, ts.values, label='Historical')
plt.plot(future_forecast.index, future_forecast.values, label='Forecast', linestyle='--', color='red', marker='o')
plt.xlabel('Date')
plt.ylabel('Titles Added')
plt.title('Netflix Content Additions: Historical + Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('forecast_plot.png', dpi=150, bbox_inches='tight')
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 5: Evaluate forecasting accuracy using the test period (Actual vs Forecast)
mae = mean_absolute_error(comparison['Actual'], comparison['Forecast'])
rmse = np.sqrt(mean_squared_error(comparison['Actual'], comparison['Forecast']))
mape = np.mean(np.abs((comparison['Actual'] - comparison['Forecast']) / comparison['Actual'])) * 100

print("\n" + "="*50)
print("FORECAST ACCURACY EVALUATION")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae:.2f} titles/month")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} titles/month")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"\nAverage actual monthly releases (test period): {comparison['Actual'].mean():.1f}")
print(f"This means our forecast is off by about {mape:.1f}% on average.")