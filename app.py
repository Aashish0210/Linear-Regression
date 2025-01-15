from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user input from form
            input_value = float(request.form['input_value'])

            # Sample data for linear regression
            X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
            y = np.array([1, 2, 1.5, 3.5, 5])

            # Linear Regression Model
            model = LinearRegression()
            model.fit(X, y)
            linear_pred = model.predict(np.array([[input_value]]))

            # Polynomial Regression Model (degree 2)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            poly_pred = poly_model.predict(poly.fit_transform(np.array([[input_value]])))

            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, color='blue', label='Data points')
            plt.plot(X, model.predict(X), color='red', label='Linear Regression Line')
            plt.plot(X, poly_model.predict(poly.fit_transform(X)), color='green', label='Polynomial Regression Line')
            plt.title('Linear vs Polynomial Regression')
            plt.xlabel('Input Value')
            plt.ylabel('Predicted Value')
            plt.legend()

            # Save the plot to a BytesIO object
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

            # Linear Regression Formula explanation
            linear_formula = r'Y = \beta_0 + \beta_1 \cdot X'
            linear_explanation = """
            In simple linear regression, we attempt to fit a straight line through the data that minimizes the error.
            The equation of the line is: Y = β₀ + β₁ * X
            where:
            - Y is the dependent variable (the predicted value)
            - X is the independent variable (the input value)
            - β₀ is the intercept (the value of Y when X is 0)
            - β₁ is the slope (how much Y changes for a unit change in X)
            """
            
            return render_template('home.html', 
                                   result=linear_pred[0], 
                                   poly_result=poly_pred[0], 
                                   img_base64=img_base64,
                                   linear_formula=linear_formula,
                                   linear_explanation=linear_explanation)

        except ValueError:
            return render_template('home.html', error="Please enter a valid number.")

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
