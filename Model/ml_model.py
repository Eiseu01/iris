import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
dataset = pd.read_csv('d:/lol/Iris/Model/Iris.csv')

# Prepare features and target
X = dataset.drop(['Species', 'Id'], axis=1)
y = dataset['Species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and scaler
pickle.dump(rf_model, open("ml_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))

# Print model accuracy
print(f"Model accuracy: {rf_model.score(X_test_scaled, y_test):.2f}")