from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["news"]
        vect = vectorizer.transform([text])
        pred = model.predict(vect)[0]
        result = "Real News" if pred == 1 else "Fake News"
    return render_template("index.html", result=result)

app.run(host="0.0.0.0", port=10000)