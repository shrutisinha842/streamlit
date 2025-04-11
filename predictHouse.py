import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('housing.csv')
X = df[['total_rooms']]  
y = df['median_house_value']  

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

if r2 < 0.3:
    interpretation = "Total Rooms is a weak predictor of Median House Value. Consider using more features."
elif r2 < 0.6:
    interpretation = "Total Rooms has a moderate relationship with Median House Value."
else:
    interpretation = "Total Rooms is a strong predictor of Median House Value."
print("Interpretation:", interpretation)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.3, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Total Rooms')
plt.ylabel('Median House Value')
plt.title('Linear Regression: Total Rooms vs Median House Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
