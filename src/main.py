import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load your dataset
data = pd.read_csv("../data/homeprices.csv")

# Feature & Label
X = data[['area']]     # must be 2D
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title(" Home Price Prediction")
plt.show()