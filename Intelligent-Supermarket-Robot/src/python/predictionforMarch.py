#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to calculate demand
def calculate_demand(unit_price, quantity, rating):
    # Assuming a linear relationship between demand, unit price, quantity, and rating
    demand = 2 * unit_price + 3 * quantity - 4 * rating 
    return demand

# Function to plot daily demand
def plot_daily_demand(df, month):
    # Filter data for the specified month
    month_df = df[df['Date'].dt.month == month].copy()
    
    # Group data by 'Product line' and 'Date'
    grouped = month_df.groupby(['Product line', 'Date'])
    
    # Initialize a dictionary to store daily demand for each product line
    daily_demand_dict = {}
    
    # Calculate demand for each group
    for (product_line, date), group_data in grouped:
        demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).sum()
        if product_line not in daily_demand_dict:
            daily_demand_dict[product_line] = {}
        daily_demand_dict[product_line][date] = demand
    
    # Plot daily demand for each product line
    plt.figure(figsize=(12, 8))
    for product_line, demand_data in daily_demand_dict.items():
        dates = list(demand_data.keys())
        demand_values = list(demand_data.values())
        plt.plot(dates, demand_values, label=product_line)

    plt.title(f'Daily Demand for Each Product Line in {month}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to calculate hypothetical daily demand for March
def calculate_hypothetical_demand(df, month):
    # Filter data for January and February
    jan_feb_df = df[df['Date'].dt.month.isin([1, 2])].copy()
    
    # Group data by 'Product line'
    grouped = jan_feb_df.groupby('Product line')
    
    # Calculate the average demand for each product line
    avg_demand_dict = {}
    for product_line, group_data in grouped:
        avg_demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).mean()
        avg_demand_dict[product_line] = avg_demand
    
    # Create a hypothetical daily demand DataFrame for March
    march_dates = pd.date_range(start='2023-03-01', end='2023-03-31', freq='D')
    march_demand_data = pd.DataFrame(index=march_dates, columns=avg_demand_dict.keys())
    
    # Fill in the hypothetical demand values
    for product_line in march_demand_data.columns:
        # Generate random variation around the average demand value
        variation = np.random.normal(loc=avg_demand_dict[product_line], scale=9, size=len(march_dates))
        march_demand_data[product_line] = avg_demand_dict[product_line] + variation
    
    return march_demand_data

# Function to prepare data for linear regression
def prepare_data_for_regression(df, march_demand_data):
    # Merge March demand data with original DataFrame
    df_regression = pd.merge(df, march_demand_data, how='left', left_on='Date', right_index=True)
    return df_regression

# Function to train linear regression model
def train_linear_regression_model(df):
    # Features for training the model
    X = df[['Unit price', 'Quantity', 'Rating']]
    
    # Target variable
    y = df['Demand']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in the features using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Initialize and train linear regression model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    return model, X_test_imputed, y_test

# Function to predict demand for March
def predict_demand_for_march(df, model):
    # Features for prediction
    features = df[['Unit price', 'Quantity', 'Rating']]
    
    # Predict demand using the trained model
    demand = model.predict(features)
    
    # Store the predicted demand in the DataFrame
    df['Demand'] = demand
    
    return df

# Path to the CSV file
csv_file_path = '/Users/cardoz/Desktop/Demand Prediction/supermarket.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Plot daily demand for January
plot_daily_demand(df, month=1)

# Plot daily demand for February
plot_daily_demand(df, month=2)

# Calculate hypothetical daily demand for March
march_demand_data = calculate_hypothetical_demand(df, month=3)

# Print total hypothetical daily demand for each product line in March
print("Total Predicted Daily Demand for Each Product Line in March:")
print(march_demand_data.sum())

# Plot hypothetical daily demand for March
plt.figure(figsize=(12, 8))
for product_line in march_demand_data.columns:
    plt.plot(march_demand_data.index, march_demand_data[product_line], label=product_line)

plt.title('Preicted Daily Demand for Each Product Line in March')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)

# Sum up hypothetical daily demand for each product line in March
total_hypothetical_demand_march = march_demand_data.sum()

# Plot total hypothetical daily demand for each product line in March as a bar graph
plt.figure(figsize=(10, 6))
total_hypothetical_demand_march.plot(kind='bar', color='skyblue')
plt.title('Total Predicted Daily Demand for Each Product Line in March')
plt.xlabel('Product Line')
plt.ylabel('Total Demand')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[18]:


# Provided data
march_hypothetical_demand = {
    'Electronic accessories': 6109.666386,
    'Fashion accessories': 6448.531543,
    'Food and beverages': 6208.761952,
    'Health and beauty': 6222.194452,
    'Home and lifestyle': 6313.152562,
    'Sports and travel': 6263.638944
}

real_demand = {
    'Electronic accessories': 5742.64,
    'Fashion accessories': 5141.02,
    'Food and beverages': 5561.54,
    'Health and beauty': 5454.66,
    'Home and lifestyle': 6192.68,
    'Sports and travel': 5658.30
}

# Calculate accuracy for each product line
accuracy_dict = {}
for product_line, hypothetical_demand in march_hypothetical_demand.items():
    real = real_demand[product_line]
    accuracy = 100 * (1 - abs((real - hypothetical_demand) / real))
    accuracy_dict[product_line] = accuracy

# Print accuracy for each product line
for product_line, accuracy in accuracy_dict.items():
    print(f"Accuracy for {product_line}: {accuracy:.2f}%")


# In[29]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to calculate demand
def calculate_demand(unit_price, quantity, rating):
    # Assuming a linear relationship between demand, unit price, quantity, and rating
    demand = 2 * unit_price + 3 * quantity - 4 * rating 
    return demand

# Function to plot daily demand
def plot_daily_demand(df, month):
    # Filter data for the specified month
    month_df = df[df['Date'].dt.month == month].copy()
    
    # Group data by 'Product line' and 'Date'
    grouped = month_df.groupby(['Product line', 'Date'])
    
    # Initialize a dictionary to store daily demand for each product line
    daily_demand_dict = {}
    
    # Calculate demand for each group
    for (product_line, date), group_data in grouped:
        demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).sum()
        if product_line not in daily_demand_dict:
            daily_demand_dict[product_line] = {}
        daily_demand_dict[product_line][date] = demand
    
    # Plot daily demand for each product line
    plt.figure(figsize=(12, 8))
    for product_line, demand_data in daily_demand_dict.items():
        dates = list(demand_data.keys())
        demand_values = list(demand_data.values())
        plt.plot(dates, demand_values, label=product_line)

    plt.title(f'Daily Demand for Each Product Line in {month}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to calculate hypothetical daily demand for March
def calculate_hypothetical_demand(df, month):
    # Filter data for January and February
    jan_feb_df = df[df['Date'].dt.month.isin([1, 2])].copy()
    
    # Group data by 'Product line'
    grouped = jan_feb_df.groupby('Product line')
    
    # Calculate the average demand for each product line
    avg_demand_dict = {}
    for product_line, group_data in grouped:
        avg_demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).mean()
        avg_demand_dict[product_line] = avg_demand
    
    # Create a hypothetical daily demand DataFrame for March
    march_dates = pd.date_range(start='2023-03-01', end='2023-03-31', freq='D')
    march_demand_data = pd.DataFrame(index=march_dates, columns=avg_demand_dict.keys())
    
    # Fill in the hypothetical demand values
    for product_line in march_demand_data.columns:
        # Generate random variation around the average demand value
        variation = np.random.normal(loc=avg_demand_dict[product_line], scale=9, size=len(march_dates))
        march_demand_data[product_line] = avg_demand_dict[product_line] + variation
    
    return march_demand_data

# Function to prepare data for linear regression
def prepare_data_for_regression(df, march_demand_data):
    # Merge March demand data with original DataFrame
    df_regression = pd.merge(df, march_demand_data, how='left', left_on='Date', right_index=True)
    return df_regression

# Function to train linear regression model
def train_linear_regression_model(df):
    # Features for training the model
    X = df[['Unit price', 'Quantity', 'Rating']]
    
    # Target variable
    y = df['Demand']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in the features using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Initialize and train linear regression model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    return model, X_test_imputed, y_test

# Function to predict demand for March
def predict_demand_for_march(df, model):
    # Features for prediction
    features = df[['Unit price', 'Quantity', 'Rating']]
    
    # Predict demand using the trained model
    demand = model.predict(features)
    
    # Store the predicted demand in the DataFrame
    df['Demand'] = demand
    
    return df

# Path to the CSV file
csv_file_path = '/Users/cardoz/Desktop/Demand Prediction/supermarket.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Plot daily demand for January
plot_daily_demand(df, month=1)

# Plot daily demand for February
plot_daily_demand(df, month=2)

# Calculate hypothetical daily demand for March
march_demand_data = calculate_hypothetical_demand(df, month=3)

# Real demand data for March
real_demand = {
    'Electronic accessories': 5742.64,
    'Fashion accessories': 5141.02,
    'Food and beverages': 5561.54,
    'Health and beauty': 5454.66,
    'Home and lifestyle': 6192.68,
    'Sports and travel': 5658.30
}

# Calculate absolute difference between predicted and real demand for each product line
absolute_difference = {product_line: abs(march_demand_data[product_line].sum() - real_demand[product_line]) 
                       for product_line in real_demand}

# Calculate total absolute difference and total demand for all product lines
total_absolute_difference = sum(absolute_difference.values())
total_real_demand = sum(real_demand.values())

# Calculate Mean Absolute Percentage Error (MAPE)
MAPE = (total_absolute_difference / total_real_demand) * 100

# Calculate Mean Absolute Error (MAE)
MAE = total_absolute_difference / len(real_demand)

# Print results
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(MAPE))
print("Mean Absolute Error (MAE): {:.2f}".format(MAE))


# In[30]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to calculate demand
def calculate_demand(unit_price, quantity, rating):
    # Assuming a linear relationship between demand, unit price, quantity, and rating
    demand = 2 * unit_price + 3 * quantity - 4 * rating 
    return demand

# Function to plot daily demand
def plot_daily_demand(df, month):
    # Filter data for the specified month
    month_df = df[df['Date'].dt.month == month].copy()
    
    # Group data by 'Product line' and 'Date'
    grouped = month_df.groupby(['Product line', 'Date'])
    
    # Initialize a dictionary to store daily demand for each product line
    daily_demand_dict = {}
    
    # Calculate demand for each group
    for (product_line, date), group_data in grouped:
        demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).sum()
        if product_line not in daily_demand_dict:
            daily_demand_dict[product_line] = {}
        daily_demand_dict[product_line][date] = demand
    
    # Plot daily demand for each product line
    plt.figure(figsize=(12, 8))
    for product_line, demand_data in daily_demand_dict.items():
        dates = list(demand_data.keys())
        demand_values = list(demand_data.values())
        plt.plot(dates, demand_values, label=product_line)

    plt.title(f'Daily Demand for Each Product Line in {month}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to calculate hypothetical daily demand for March
def calculate_hypothetical_demand(df, month):
    # Filter data for January and February
    jan_feb_df = df[df['Date'].dt.month.isin([1, 2])].copy()
    
    # Group data by 'Product line'
    grouped = jan_feb_df.groupby('Product line')
    
    # Calculate the average demand for each product line
    avg_demand_dict = {}
    for product_line, group_data in grouped:
        avg_demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).mean()
        avg_demand_dict[product_line] = avg_demand
    
    # Create a hypothetical daily demand DataFrame for March
    march_dates = pd.date_range(start='2023-03-01', end='2023-03-31', freq='D')
    march_demand_data = pd.DataFrame(index=march_dates, columns=avg_demand_dict.keys())
    
    # Fill in the hypothetical demand values
    for product_line in march_demand_data.columns:
        # Generate random variation around the average demand value
        variation = np.random.normal(loc=avg_demand_dict[product_line], scale=9, size=len(march_dates))
        march_demand_data[product_line] = avg_demand_dict[product_line] + variation
    
    return march_demand_data

# Function to prepare data for linear regression
def prepare_data_for_regression(df, march_demand_data):
    # Merge March demand data with original DataFrame
    df_regression = pd.merge(df, march_demand_data, how='left', left_on='Date', right_index=True)
    return df_regression

# Function to train linear regression model
def train_linear_regression_model(df):
    # Features for training the model
    X = df[['Unit price', 'Quantity', 'Rating']]
    
    # Target variable
    y = df['Demand']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in the features using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Initialize and train linear regression model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    return model, X_test_imputed, y_test

# Function to predict demand for March
def predict_demand_for_march(df, model):
    # Features for prediction
    features = df[['Unit price', 'Quantity', 'Rating']]
    
    # Predict demand using the trained model
    demand = model.predict(features)
    
    # Store the predicted demand in the DataFrame
    df['Demand'] = demand
    
    return df

# Path to the CSV file
csv_file_path = '/Users/cardoz/Desktop/Demand Prediction/supermarket.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Plot daily demand for January
plot_daily_demand(df, month=1)

# Plot daily demand for February
plot_daily_demand(df, month=2)

# Calculate hypothetical daily demand for March
march_demand_data = calculate_hypothetical_demand(df, month=3)

# Real demand data for March
real_demand = {
    'Electronic accessories': 5742.64,
    'Fashion accessories': 5141.02,
    'Food and beverages': 5561.54,
    'Health and beauty': 5454.66,
    'Home and lifestyle': 6192.68,
    'Sports and travel': 5658.30
}

# Calculate absolute difference between predicted and real demand for each product line
absolute_difference = {product_line: abs(march_demand_data[product_line].sum() - real_demand[product_line]) 
                       for product_line in real_demand}

# Calculate total absolute difference and total demand for all product lines
total_absolute_difference = sum(absolute_difference.values())
total_real_demand = sum(real_demand.values())

# Calculate Mean Absolute Percentage Error (MAPE)
MAPE = (total_absolute_difference / total_real_demand) * 100

# Calculate Mean Absolute Error (MAE)
MAE = total_absolute_difference / len(real_demand)

# Print results
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(MAPE))
print("Mean Absolute Error (MAE): {:.2f}".format(MAE))

# Print total predicted demand for each product line in March
print("Total Predicted Daily Demand for Each Product Line in March:")
print(march_demand_data.sum())

# Plot hypothetical daily demand for March
plt.figure(figsize=(12, 8))
for product_line in march_demand_data.columns:
    plt.plot(march_demand_data.index, march_demand_data[product_line], label=product_line)

plt.title('Predicted Daily Demand for Each Product Line in March')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Sum up hypothetical daily demand for each product line in March
total_hypothetical_demand_march = march_demand_data.sum()

# Plot total hypothetical daily demand for each product line in March as a bar graph
plt.figure(figsize=(10, 6))
total_hypothetical_demand_march.plot(kind='bar', color='skyblue')
plt.title('Total Predicted Daily Demand for Each Product Line in March')
plt.xlabel('Product Line')
plt.ylabel('Total Demand')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[34]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to calculate demand
def calculate_demand(unit_price, quantity, rating):
    # Assuming a linear relationship between demand, unit price, quantity, and rating
    demand = 2 * unit_price + 3 * quantity - 4 * rating 
    return demand

# Function to plot daily demand
def plot_daily_demand(df, month):
    # Filter data for the specified month
    month_df = df[df['Date'].dt.month == month].copy()
    
    # Group data by 'Product line' and 'Date'
    grouped = month_df.groupby(['Product line', 'Date'])
    
    # Initialize a dictionary to store daily demand for each product line
    daily_demand_dict = {}
    
    # Calculate demand for each group
    for (product_line, date), group_data in grouped:
        demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).sum()
        if product_line not in daily_demand_dict:
            daily_demand_dict[product_line] = {}
        daily_demand_dict[product_line][date] = demand
    
    # Plot daily demand for each product line
    plt.figure(figsize=(12, 8))
    for product_line, demand_data in daily_demand_dict.items():
        dates = list(demand_data.keys())
        demand_values = list(demand_data.values())
        plt.plot(dates, demand_values, label=product_line)

    plt.title(f'Daily Demand for Each Product Line in {month}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to calculate hypothetical daily demand for March
def calculate_hypothetical_demand(df, month):
    # Filter data for January and February
    jan_feb_df = df[df['Date'].dt.month.isin([1, 2])].copy()
    
    # Group data by 'Product line'
    grouped = jan_feb_df.groupby('Product line')
    
    # Calculate the average demand for each product line
    avg_demand_dict = {}
    for product_line, group_data in grouped:
        avg_demand = calculate_demand(group_data['Unit price'], group_data['Quantity'], group_data['Rating']).mean()
        avg_demand_dict[product_line] = avg_demand
    
    # Create a hypothetical daily demand DataFrame for March
    march_dates = pd.date_range(start='2023-03-01', end='2023-03-31', freq='D')
    march_demand_data = pd.DataFrame(index=march_dates, columns=avg_demand_dict.keys())
    
    # Fill in the hypothetical demand values
    for product_line in march_demand_data.columns:
        # Generate random variation around the average demand value
        variation = np.random.normal(loc=avg_demand_dict[product_line], scale=9, size=len(march_dates))
        march_demand_data[product_line] = avg_demand_dict[product_line] + variation
    
    return march_demand_data

# Function to prepare data for linear regression
def prepare_data_for_regression(df, march_demand_data):
    # Merge March demand data with original DataFrame
    df_regression = pd.merge(df, march_demand_data, how='left', left_on='Date', right_index=True)
    return df_regression

# Function to train linear regression model
def train_linear_regression_model(df):
    # Features for training the model
    X = df[['Unit price', 'Quantity', 'Rating']]
    
    # Target variable
    y = df['Demand']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in the features using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Initialize and train linear regression model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    return model, X_test_imputed, y_test

# Function to predict demand for March
def predict_demand_for_march(df, model):
    # Features for prediction
    features = df[['Unit price', 'Quantity', 'Rating']]
    
    # Predict demand using the trained model
    demand = model.predict(features)
    
    # Store the predicted demand in the DataFrame
    df['Demand'] = demand
    
    return df

# Path to the CSV file
csv_file_path = '/Users/cardoz/Desktop/Demand Prediction/supermarket.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Plot daily demand for January
plot_daily_demand(df, month=1)

# Plot daily demand for February
plot_daily_demand(df, month=2)

# Calculate hypothetical daily demand for March
march_demand_data = calculate_hypothetical_demand(df, month=3)

# Real demand data for March
real_demand = {
    'Electronic accessories': 5742.64,
    'Fashion accessories': 5141.02,
    'Food and beverages': 5561.54,
    'Health and beauty': 5454.66,
    'Home and lifestyle': 6192.68,
    'Sports and travel': 5658.30
}

# Calculate absolute difference between predicted and real demand for each product line
absolute_difference = {product_line: abs(march_demand_data[product_line].sum() - real_demand[product_line]) 
                       for product_line in real_demand}

# Calculate total absolute difference and total demand for all product lines
total_absolute_difference = sum(absolute_difference.values())
total_real_demand = sum(real_demand.values())

# Calculate Mean Absolute Percentage Error (MAPE)
MAPE = (total_absolute_difference / total_real_demand) * 100

# Calculate Mean Absolute Error (MAE)
MAE = total_absolute_difference / len(real_demand)

# Calculate accuracy for each product line
accuracy = {product_line: 1 - (absolute_difference[product_line] / real_demand[product_line]) for product_line in real_demand}

# Print results
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(MAPE))
print("Mean Absolute Error (MAE): {:.2f}".format(MAE))
print("Accuracy for Each Product Line:")
for product_line, acc in accuracy.items():
    print(f"{product_line}: {acc:.2%}")

# Print total predicted demand for each product line in March
print("\nTotal Predicted Daily Demand for Each Product Line in March:")
print(march_demand_data.sum())

# Plot hypothetical daily demand for March
plt.figure(figsize=(12, 8))
for product_line in march_demand_data.columns:
    plt.plot(march_demand_data.index, march_demand_data[product_line], label=product_line)

plt.title('Predicted Daily Demand for Each Product Line in March')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Sum up hypothetical daily demand for each product line in March
total_hypothetical_demand_march = march_demand_data.sum()

# Plot total hypothetical daily demand for each product line in March as a bar graph
plt.figure(figsize=(10, 6))
total_hypothetical_demand_march.plot(kind='bar', color='skyblue')
plt.title('Total Predicted Daily Demand for Each Product Line in March')
plt.xlabel('Product Line')
plt.ylabel('Total Demand')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[ ]:




