# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("forex-1year.csv")
    print("Data loaded successfully!")
    print("First 5 rows of the dataset")
    print(df.head().to_string())
except FileNotFoundError:
    print("Error: The file 'forex-1year.csv' was not found. Please ensure it is in the same directory.")
    exit()

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing values")
print(df_missing)

# Check for duplicate rows
df_duplicated = df.duplicated().sum()
print("Duplcated rows")
print(df_duplicated)

# Define the features (X) and the target (y). For multiple linear regression
# we include all relevant features in our feature set
X = df[["open_eurusd","high_eurusd","low_eurusd",]] # Independent variables (feattures)
y = df["close_eurusd"] # Dependent variable (target)

print("Shape of features (X):",X.shape)
print("shape of target (y):",y.shape)

# %% [markdown]
# Data Splitting

# %%
# We split the data into a training set and testing set. The model learns from the 
# training data and then is evaluated on the testing data, which it has not seen before
# We'll use the 70/30 split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Number of samples in training set:",len(X_train))
print("Number of samples in testing set:",len(X_test))

# %% [markdown]
# Model Training

# %%
# We create an instance of the Linear Regression model and train it using the
# training data. The 'fit' method finds the best-fit linear equation that represents
# the relationship between all the input features and the target.
print("\nTraining the Multiple Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")


# %% [markdown]
# Model Evaluation

# %%
# Now we make a predictions on the test set and evaluate and the model's performance
# We'll use the Mean Squared Error (MSE) and R-squared to measure accuracy
# A higher R-squared value indicates a better fit
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# %% [markdown]
# Making a New Prediction 

# %%
# This part of the code prompts the user to input values for all features to make a prediction
print("Enter the following values to predict the Close Price")

while True:
    try:
        open_price = float(input("Enter the Open price (or type 'exit' to quit):"))
        if open_price is None:
            break
        high_price = float(input("Enter the High Price: "))
        low_price = float(input("Enter the Low price: "))

        # We must reshape the input to a 2D array, even for a single sample
        # The order of the features must match the order used during training
        new_prices = np.array([[open_price,high_price,low_price]])

        predicted_close = model.predict(new_prices)

        print(f"For the given prices, the predicted Close price is {predicted_close[0]:.4f}")
    except ValueError:
        if input == "exit":
            break
        print("Invalid input. Please enter valid numbers for all three fields.")


