Implementation Code (Python)


The following Python implementation covers data loading, preprocessing, model training, evaluation, and prediction. The code is modular and fully reproducible.

6.1 Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

6.2 Loading and Exploring the Dataset
# Load dataset
df = pd.read_csv('crop_yield.csv')
print('Shape:', df.shape)
print(df.head())
print(df.info())
print(df.describe())
print('Missing values:')
print(df.isnull().sum())

6.3 Data Preprocessing
# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
le_crop = LabelEncoder()
le_country = LabelEncoder()
df['Crop_Encoded']    = le_crop.fit_transform(df['Item'])
df['Country_Encoded'] = le_country.fit_transform(df['Area'])

# Select features and target
features = ['Crop_Encoded', 'Country_Encoded',
            'average_rain_fall_mm_per_year',
            'pesticides_tonnes',
            'avg_temp']
X = df[features]
y = df['hg/ha_yield']

6.4 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Training samples : {X_train.shape[0]}')
print(f'Testing  samples : {X_test.shape[0]}')

6.5 Model Training
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display coefficients
print('Intercept:', model.intercept_)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print(coef_df)

6.6 Model Evaluation
# Generate predictions
y_pred = model.predict(X_test)

# Compute metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f'MAE  : {mae:.2f}')
print(f'MSE  : {mse:.2f}')
print(f'RMSE : {rmse:.2f}')
print(f'R2   : {r2:.4f}')

6.7 Visualization
# --- Actual vs Predicted ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='green', edgecolors='k', s=30)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield (hg/ha)')
plt.ylabel('Predicted Yield (hg/ha)')
plt.title('Actual vs Predicted Crop Yield')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# --- Residual Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Yield')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['hg/ha_yield']].corr(),
            annot=True, cmap='YlGn', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

6.8 Making a Prediction
# Predict yield for a new sample
# Example: Wheat crop, India, 1200mm rain, 50 tonnes pesticide, 25°C
crop_enc    = le_crop.transform(['Wheat'])[0]
country_enc = le_country.transform(['India'])[0]

new_input = pd.DataFrame([[crop_enc, country_enc, 1200, 50, 25]],
                          columns=features)
prediction = model.predict(new_input)
print(f'Predicted Yield: {prediction[0]:.2f} hg/ha')
