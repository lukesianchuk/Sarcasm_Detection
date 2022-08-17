from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        cv = joblib.load("cv.pkl")
        
        # Get values through input bars
        sentence = request.form.get("sentence")
        
        # transform inputs
        trans = cv.transform([sentence]).toarray()
        
        
        # Get prediction
        prediction = clf.predict(trans)[0]
        probability = clf.predict_proba(trans)
        probability = (max(probability[0]))
        
        
    else:
        prediction = ""
        probability = ""
        
    return render_template("website.html", pred = prediction, prob = probability)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='0.0.0.0', port=5000)

