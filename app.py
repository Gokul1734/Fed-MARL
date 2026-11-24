from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pickle
import numpy as np
from twilio.rest import Client

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# Reverse mapping dictionary (make sure it matches your training)
crop_dict = {
    0: 'rice',
    1: 'maize',
    2: 'jute',
    3: 'cotton',
    4: 'coconut',
    5: 'papaya',
    6: 'orange',
    7: 'apple',
    8: 'muskmelon',
    9: 'watermelon',
    10: 'grapes',
    11: 'mango',
    12: 'banana',
    13: 'pomegranate',
    14: 'lentil',
    15: 'blackgram',
    16: 'mungbean',
    17: 'mothbeans',
    18: 'pigeonpeas',
    19: 'kidneybeans',
    20: 'chickpea',
    21: 'coffee'
}


# Twilio credentials
TWILIO_ACCOUNT_SID = "ACaf10644b235062930d3d62c1dc13d9ef"
TWILIO_AUTH_TOKEN = "3ac33df95c2a8f198ff2e136379ce3b9"
TWILIO_PHONE = "+19036622511"  
# ================= Database Setup ==================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT, phone TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, username TEXT, N REAL, P REAL, K REAL,
                  temperature REAL, humidity REAL, moisture REAL, crop TEXT)''')
    conn.commit()
    conn.close()

init_db()


# ================= Routes ==================
@app.route("/")
def home():
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        phone = request.form["phone"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, phone) VALUES (?, ?, ?)",
                  (username, password, phone))
        conn.commit()
        conn.close()
        flash("Account created successfully! Please login.")
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        session["username"] = username
        return redirect(url_for("predict"))
    else:
        flash("Invalid login credentials")
        return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])

        features = np.array([[N, P, K, temperature, humidity,moisture]])
        prediction_index = model.predict(features)[0]
        prediction = crop_dict[prediction_index]  
        # Save in DB
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO predictions (username, N, P, K, temperature, humidity, moisture, crop) VALUES (?,?,?,?,?,?,?,?)",
                  (session["username"], N, P, K, temperature, humidity,moisture, prediction))
        conn.commit()
        conn.close()

        # Fetch user phone
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT phone FROM users WHERE username=?", (session["username"],))
        phone = c.fetchone()[0]
        conn.close()

        # Send SMS using Twilio
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Hello {session['username']}, the best crop for your field is {prediction}.",
            from_=TWILIO_PHONE,
            to="+919791970373"
        )

        return render_template("result.html", crop=prediction)

    return render_template("predict.html")



@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
