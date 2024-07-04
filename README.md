
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
spotify_data = pd.read_csv('spotify_dataset.csv')

# Check for missing values
missing_values = spotify_data.isnull().sum()
print("Missing values:\n", missing_values)

# Normalize or scale the data if required
scaler = StandardScaler()

# Drop non-numeric columns or columns not needed for analysis
spotify_data_numeric = spotify_data.drop(['track_id', 'track_name', 'artist_name', 'playlist_name', 'playlist_genre'], axis=1)

# Handle missing values if any (if necessary)
# For example: spotify_data_numeric.fillna(method='ffill', inplace=True)

# Scale the numeric features
scaled_features = scaler.fit_transform(spotify_data_numeric)

# Correlation Matrix
correlation_matrix = spotify_data_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Clustering

kmeans = KMeans(n_clusters=5, random_state=42)
spotify_data['cluster'] = kmeans.fit_predict(scaled_features)



# Model Building
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, spotify_data['playlist_genre'], test_size=0.2, random_state=42)

# Build the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
