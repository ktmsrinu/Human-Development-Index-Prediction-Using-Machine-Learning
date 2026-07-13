import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("hdi_dataset.csv")

X = df[[
'Life Expectancy at Birth, male (2021)',
'Mean Years of Schooling, male (2021)',
'Expected Years of Schooling, male (2021)',
'Gross National Income Per Capita (2021)'
]]

y = df['Planetary pressures-adjusted Human Development Index (2021)']

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

print("R2 Score:", score)
import pickle

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")