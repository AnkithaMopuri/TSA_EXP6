# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16-05-2026

### NAME: MOPURI ANKITHA
### REGISTER NUMBER: 212223040117

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

# For normal plots
%matplotlib inline
plt.style.use('default')

# Step 1: Load dataset
data = pd.read_csv('movies.csv')

# Step 2: Convert release_date to datetime
data['release_date'] = pd.to_datetime(data['release_date'], dayfirst=True)

# Step 3: Sort values by date
data = data.sort_values('release_date')

# Step 4: Set release_date as index
data.set_index('release_date', inplace=True)

# Step 5: Use vote_average column and resample monthly
data_monthly = data['vote_average'].resample('MS').mean()

# Step 6: Fill missing values
data_monthly = data_monthly.fillna(method='ffill')

# Step 7: Set monthly frequency
data_monthly = data_monthly.asfreq('MS')

# Step 8: Display first rows
print(data_monthly.head())

# Step 9: Plot original monthly data
plt.figure(figsize=(12,6))

data_monthly.plot()

plt.title('Monthly Movie Vote Average')

plt.xlabel('Release Date')

plt.ylabel('Vote Average')

plt.grid(True)

plt.show()

# Step 10: Plot data
plt.figure(figsize=(12,6))

data_monthly.plot()

plt.title('Movie Vote Average Data')

plt.xlabel('Release Date')

plt.ylabel('Vote Average')

plt.grid(True)

plt.show()

# Step 11: Seasonal decomposition
decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

decomposition.plot()

plt.show()

# Step 12: Add 1 for multiplicative seasonality
data_monthly = data_monthly + 1

# Step 13: Split train and test data
train_size = int(len(data_monthly) * 0.8)

train_data = data_monthly[:train_size]

test_data = data_monthly[train_size:]

# Step 14: Create Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Step 15: Predict test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Step 16: Plot train, test and predictions
plt.figure(figsize=(12,6))

ax = train_data.plot()

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend([
    "Train Data",
    "Predicted Data",
    "Test Data"
])

ax.set_title('Holt-Winters Prediction Evaluation')

plt.grid(True)

plt.show()

# Step 17: Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

# Step 18: Print RMSE
print("Root Mean Squared Error (RMSE):", rmse)

# Step 19: Print Mean and Standard Deviation
print("Mean:", data_monthly.mean())

print("Standard Deviation:", data_monthly.std())

# Step 20: Final model using full dataset
final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Step 21: Forecast future values
future_steps = int(len(data_monthly) / 4)

final_predictions = final_model.forecast(steps=future_steps)

# Step 22: Plot future predictions
plt.figure(figsize=(12,6))

ax = data_monthly.plot()

final_predictions.plot(ax=ax)

ax.legend([
    "Original Data",
    "Future Predictions"
])

ax.set_xlabel('Release Date')

ax.set_ylabel('Vote Average')

ax.set_title('Future Movie Prediction using Holt-Winters Method')

plt.grid(True)

plt.show()
```

### OUTPUT:
<img width="1001" height="545" alt="image" src="https://github.com/user-attachments/assets/722f4e7c-0f7a-4db0-a818-58135906867d" />
<br>

# Decomposed Plot:

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/0b986564-5fbd-4659-b0f6-9c2de64d0fbc" />
<br>



# TEST_PREDICTION
<img width="981" height="545" alt="image" src="https://github.com/user-attachments/assets/2052a95a-9b5d-40b9-9666-eb78799df973" />


# FINAL_PREDICTION
<img width="1001" height="545" alt="image" src="https://github.com/user-attachments/assets/5669d806-b8dc-492d-9f5b-40d6fa0c9b0a" />

<img width="1138" height="110" alt="image" src="https://github.com/user-attachments/assets/86ed2294-a116-4d9e-970f-ecea5dae2ccc" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
