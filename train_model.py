from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load IRIS data
iris = load_iris()
X = iris.data
y = iris.target

# Train a simple model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model to a file
joblib.dump(model, 'model/iris_model.pkl')
print("Model saved in model/iris_model.pkl")
