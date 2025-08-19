import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("C:/Users/bhavyanshu/Desktop/New folder (2)/Dataset/kidney_disease.csv")

# Drop ID column
data = data.drop("id", axis=1)

# Convert numeric object columns to float
for col in ['pcv', 'wc', 'rc']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Clean target labels
data['classification'] = data['classification'].replace('ckd\t', 'ckd')
data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})

# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data.drop("classification", axis=1)
y = data["classification"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("kidney_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'kidney_model.pkl'")
