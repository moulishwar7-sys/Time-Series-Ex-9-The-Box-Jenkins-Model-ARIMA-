# Time-Series-Ex-9-The-Box-Jenkins-Model-ARIMA-

# AIM:
To understand and implement the Box-Jenkins approach to time-series forecasting using an ARIMA model in Python.

# PROGRAM:
```
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

sales = [
 100, 105, 103, 110, 115, 112, 120, 125,
 123, 130, 135, 132, 140, 145, 143, 150,
 155, 152, 160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("STEP 1: Original Sales Data")
print(data.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))
if adf_result[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Insufficient evidence of stationarity.")
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))
plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))
if adf_diff[1] < 0.05:
 print("Conclusion: Evidence of stationarity.")
else:
 print("Conclusion: Stationarity is not confirmed.")

print("\nSTEP 5: ACF and PACF Plots")
plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()
plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))
residuals = fitted_model.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()
lb_result = acorr_ljungbox(residuals, lags=[5], return_df=True)
print("\nLjung-Box Test Result")
print(lb_result)
if lb_result["lb_pvalue"].iloc[0] > 0.05:
 print("No significant residual autocorrelation detected at lag 5.")
else:
 print("Residual autocorrelation may remain; review the model.")
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")
for month, value in enumerate(forecast_mean, start=len(data) + 1):
 print(f"Month {month}: {value:.2f}")
print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))
future_months = range(len(data), len(data) + 6)
plt.figure(figsize=(10, 5))
plt.plot(range(len(data)), data, marker="o", label="Actual Sales")
plt.plot(future_months, forecast_mean.to_numpy(), marker="o", linestyle="--",
label="Forecast")
plt.fill_between(
 future_months,
 forecast_ci.iloc[:, 0].to_numpy(),
 forecast_ci.iloc[:, 1].to_numpy(),
 alpha=0.2,
 label="95% Confidence Interval"
)
plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed successfully.")
```

# OUTPUT:
<img width="606" height="267" alt="image" src="https://github.com/user-attachments/assets/065922bd-b651-44ed-a6cf-daf4eb96a380" />

<img width="631" height="560" alt="image" src="https://github.com/user-attachments/assets/2e92890a-0f4e-4fcf-9cf8-3129825db9cf" />

<img width="632" height="263" alt="image" src="https://github.com/user-attachments/assets/832b2ff3-9092-47e8-9eab-fcdb27a8ebf9" />

<img width="628" height="491" alt="image" src="https://github.com/user-attachments/assets/9d3e9f7a-8c9a-4b13-8996-e4dd87f29ad9" />

<img width="635" height="446" alt="image" src="https://github.com/user-attachments/assets/7fcbe7fc-4671-45b0-9616-5f17fec861fe" />

<img width="615" height="505" alt="image" src="https://github.com/user-attachments/assets/48c02fc5-faa6-4766-b379-c314a814d772" />

<img width="608" height="322" alt="image" src="https://github.com/user-attachments/assets/66383687-4b17-4a02-8d3d-abdd76ac7e2a" />

<img width="486" height="282" alt="image" src="https://github.com/user-attachments/assets/0e7f58bd-7a00-41cb-852f-57747c07e9f5" />

<img width="641" height="287" alt="image" src="https://github.com/user-attachments/assets/5c467c05-fd0f-4eb2-9d04-620887495ca1" />


























