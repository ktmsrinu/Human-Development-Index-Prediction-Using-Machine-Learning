from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    life = float(request.form["life"])
    school = float(request.form["school"])
    expected = float(request.form["expected"])
    income = float(request.form["income"])

    result = model.predict([[life, school, expected, income]])

    pred = result[0]

    if pred >= 0.800:
        category = "Very High HDI"
    elif pred >= 0.700:
        category = "High HDI"
    elif pred >= 0.550:
        category = "Medium HDI"
    else:
        category = "Low HDI"

    return render_template(
        "index.html",
        prediction=round(pred, 4),
        category=category
    )

if __name__ == "__main__":
    app.run(debug=True)