import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



data = pd.read_csv('loan.csv')


label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['occupation'] = label_encoder.fit_transform(data['occupation'])
data['education_level'] = label_encoder.fit_transform(data['education_level'])
data['marital_status'] = label_encoder.fit_transform(data['marital_status'])
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])


X = data.drop('loan_status', axis=1)
y = data['loan_status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


# Plot training points
plt.scatter(X_train['age'], X_train['income'], c=y_train, cmap='coolwarm', label='Training Data')

# Plot prediction points
plt.scatter(X_test['age'], X_test['income'], c=y_pred, cmap='coolwarm', marker='x', label='Prediction Data')

plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Loan Qualification')
plt.legend()
plt.show()

# Get feature importances from the trained decision tree model
feature_importances = clf.feature_importances_
top_three_indices = np.argsort(feature_importances)[::-1][:3]
top_three_features = X.columns[top_three_indices]

# Create a 3D scatter plot using the top three features
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[top_three_features[0]], X_train[top_three_features[1]], X_train[top_three_features[2]], c=y_train, cmap='coolwarm', label='Training Data')

# Plot prediction points
ax.scatter(X_test[top_three_features[0]], X_test[top_three_features[1]], X_test[top_three_features[2]], c=y_pred, cmap='coolwarm', marker='x', label='Prediction Data')

ax.set_xlabel(top_three_features[0])
ax.set_ylabel(top_three_features[1])
ax.set_zlabel(top_three_features[2])
ax.set_title('3D Scatter Plot of Top Three Features')
ax.legend()

plt.show()