from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/iris_model.pkl')
iris_labels = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/', methods=['GET', 'POST'])
def home():
    # Set default values for form and result
    prediction_text = None
    image_path = None
    input_values = {
        'sepal_length': '',
        'sepal_width': '',
        'petal_length': '',
        'petal_width': ''
    }

    if request.method == 'POST':
        try:
            # Get form data
            input_values['sepal_length'] = request.form['sepal_length']
            input_values['sepal_width'] = request.form['sepal_width']
            input_values['petal_length'] = request.form['petal_length']
            input_values['petal_width'] = request.form['petal_width']

            # Convert inputs to float
            features = [
                float(input_values['sepal_length']),
                float(input_values['sepal_width']),
                float(input_values['petal_length']),
                float(input_values['petal_width'])
            ]

            # Make prediction
            prediction = model.predict([features])[0]
            result = iris_labels[prediction]
            prediction_text = f'Predicted Iris Class: {result}'

            # Choose image based on prediction
            image_path = f'static/{result.lower()}.jpg'

        except Exception as e:
            prediction_text = 'Invalid input. Please enter valid numbers.'

    return render_template('index.html', prediction_text=prediction_text, input_values=input_values, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)




