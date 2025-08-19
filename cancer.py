import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Dataset/cancer.csv")

# Drop unnecessary columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Encode target variable (M=1, B=0)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model to file
with open("cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'cancer_model.pkl'")
