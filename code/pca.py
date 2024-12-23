import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the dataset
file_path = r'C:\Users\chide\OneDrive\Desktop\infosys\Dataset\dynamic_pricing.csv'
data = pd.read_csv(file_path)

# Display the first few rows and column information
data.head(), data.info()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Selecting the features to scale
features_to_scale = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features_to_scale])

# Converting scaled features back to DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=features_to_scale)

# Applying PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Calculating the explained variance for each component
explained_variance = pca.explained_variance_ratio_

# Creating a DataFrame to display the PCA results and explained variance
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(len(features_to_scale))])
explained_variance_df = pd.DataFrame(explained_variance, index=[f'PC{i+1}' for i in range(len(features_to_scale))],
                                     columns=['Explained Variance Ratio'])

# Display PCA results and explained variance
pca_df.head(), explained_variance_df
from sklearn.model_selection import train_test_split

# Defining the features (scaled features from PCA) and target variable
X = pca_df.values  # PCA transformed features
y = data['Historical_Cost_of_Ride'].values  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display the shapes of the train and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Standardizing the training and test sets again for final PCA
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Applying PCA to reduce to 2 components
pca_final = PCA(n_components=2)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

# Explained variance for the selected components
explained_variance_final = pca_final.explained_variance_ratio_

# Fitting Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(X_train_pca, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test_pca)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Scatter plot: Predicted vs Actual for test data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Actual Cost of Ride')
plt.ylabel('Predicted Cost of Ride')
plt.title('Predicted vs Actual Cost of Ride (Test Set)')
plt.show()

# Explained variance and performance metrics output
explained_variance_final, mse, r2

