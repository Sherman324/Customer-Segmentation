import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#load dataset
data = pd.read_csv("data/Test.csv")

# List of columns to drop
columns_to_drop = ["ID", "Gender", "Ever_Married", "Age", "Graduated", "Work_Experience", "Var_1", "Profession"]

# Drop the specified columns
data = data.drop(columns=columns_to_drop)

# Display the first few rows of the updated DataFrame
print(data.head())

# map segmentation and spending score to numerical values
data["Spending_Score"] = data["Spending_Score"].map({
   "Low": 3 ,
   "Average": 2 ,
   "High": 1 
})

#print(data["Spending_Score"].value_counts())

data["Segmentation"] = data["Segmentation"].map({
   "A": 1 ,
   "B": 2 ,
   "C": 3 , 
   "D": 4
})

#print(data["Segmentation"].value_counts())

# Define features and labels (X and Y)
X = data.drop(columns="Segmentation")  
Y = data["Segmentation"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

# Initialize and train the model
model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(y_pred)

#model eveluation
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Save the model
joblib.dump(model, filename='model/model.pkl')