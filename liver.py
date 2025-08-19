import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Dataset/liver.csv")

# Fill missing values in 'Albumin_and_Globulin_Ratio'
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

# Encode 'Gender' column
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# Map 'Dataset' column: 1 = liver disease, 2 = no liver disease
data['Dataset'] = data['Dataset'].map({1: 1, 2: 0})

# Split features and target
X = data.drop('Dataset', axis=1)
y = data['Dataset']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("liver_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'liver_model.pkl'")
