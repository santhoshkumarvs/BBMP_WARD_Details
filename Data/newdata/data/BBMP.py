import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('eco.csv')
features = ['male_population', 'female_population', 'ward_Name', 'distance']
target = 'Total_population'
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
# Train a Linear Regression model
model = LinearRegression()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['ward_Name'] = le.fit_transform(data['ward_Name'])

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
model.fit(X_train, y_train)



# Make predictions on the testing data
y_pred = model.predict(X_test)


# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)


print(f"Mean Absolute Error: {mae}")




import numpy as np


# Get the maximum encoded value
max_encoded_value = np.max(le.transform(le.classes_))

# Manually encode the new value
new_ward_encoded = max_encoded_value + 1

# Append the new value to the classes_
le.classes_ = np.append(le.classes_, "New_Ward")


import pandas as pd


df = pd.read_csv('eco.csv', header=0)

print(df.columns)

df['total_population'] = df['male_population'] + df['female_population']



print(df)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['male_population', 'female_population']], df['total_population'], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)


y_pred = model.predict(X_test)







print(y_pred)
