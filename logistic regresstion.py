import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
#  Load dataset

df = pd.read_csv("E:\\py\\random project\\lgr.py\\predictive_maintenance.csv")  # Update with your actual path
df.columns = df.columns.str.strip() # Remove any leading/trailing whitespace from column names
#  Separate features and target
X = df[
    ["Air temperature [K]",
     "Process temperature [K]",
     "Rotational speed [rpm]",
     "Torque [Nm]",
     "Tool wear [min]"]
]

y = df["Target"]  # 0 or 1

#  Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # 80% train, 20% test, random state for reproducibility (here im using 42 but you can choose any number)
)

#  Create model
model = LogisticRegression()

#  Train model
model.fit(X_train, y_train)

#  Make predictions
y_pred = model.predict(X_test)

#  Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Predict probability of failure   
air_temperature = float(input("Enter Air temperature [K]: "))
process_temperature = float(input("Enter Process temperature [K]: ")) 
rotational_speed = float(input("Enter Rotational speed [rpm]: "))
torque = float(input("Enter Torque [Nm]: "))
tool_wear = float(input("Enter Tool wear [min]: "))
new_machine = pd.DataFrame([{
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])
proba = model.predict_proba(new_machine)
probability = proba[0][1] * 100
print(f"Failure probability: {probability:.2f}%")