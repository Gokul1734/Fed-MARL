"""
Enhanced Flask Web Application
Integrates all Smart Agriculture modules with comprehensive UI
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import pickle
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from crop_recommendation import CropRecommendationSystem
    from crop_yield_prediction import CropYieldPredictor
    from disease_detection import DiseaseRiskDetector
    from irrigation_recommendation import IrrigationRecommendationSystem
except ImportError:
    print(" Custom modules not found. Make sure to train models first using train_models.py")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smart-agriculture-secret-key-2024')

# Global model variables
crop_recommender = None
yield_predictor = None
disease_detector = None
irrigation_system = None
preprocessor = None

# Load models on startup
def load_models():
    """Load all trained ML models"""
    global crop_recommender, yield_predictor, disease_detector, irrigation_system, preprocessor
    
    print("Loading models...")
    
    try:
        # Load crop recommendation model
        if os.path.exists('models/crop_recommendation_model.pkl'):
            crop_recommender = CropRecommendationSystem()
            crop_recommender.load_model('models/crop_recommendation_model.pkl')
            print(" Crop recommendation model loaded")
        
        # Load yield prediction model
        if os.path.exists('models/yield_prediction_model.pkl'):
            yield_predictor = CropYieldPredictor()
            yield_predictor.load_model('models/yield_prediction_model.pkl')
            print(" Yield prediction model loaded")
        
        # Load disease detection model
        if os.path.exists('models/disease_detection_model.pkl'):
            disease_detector = DiseaseRiskDetector()
            disease_detector.load_model('models/disease_detection_model.pkl')
            print(" Disease detection model loaded")
        
        # Initialize irrigation system
        irrigation_system = IrrigationRecommendationSystem()
        print(" Irrigation system initialized")
        
        # Load preprocessor
        if os.path.exists('models/preprocessor.pkl'):
            with open('models/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            print(" Preprocessor loaded")
        
        print(" All models loaded successfully")
        return True
        
    except Exception as e:
        print(f" Error loading models: {e}")
        print(" Please train models first using: python train_models.py")
        return False


# Initialize database
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect("agriculture_database.db")
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  phone TEXT,
                  email TEXT,
                  farm_location TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Crop predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS crop_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  N REAL, P REAL, K REAL,
                  temperature REAL, humidity REAL, ph REAL, rainfall REAL,
                  recommended_crop TEXT,
                  confidence REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Yield predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS yield_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  crop TEXT,
                  N REAL, P REAL, K REAL,
                  temperature REAL, humidity REAL, ph REAL, rainfall REAL,
                  predicted_yield REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Disease predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS disease_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  N REAL, P REAL, K REAL,
                  temperature REAL, humidity REAL, ph REAL, rainfall REAL,
                  risk_level TEXT,
                  confidence REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Irrigation recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS irrigation_recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  crop TEXT,
                  soil_moisture REAL,
                  temperature REAL, humidity REAL, rainfall_7days REAL,
                  irrigation_needed BOOLEAN,
                  amount_mm REAL,
                  urgency TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()
    print(" Database initialized")


# ==================== ROUTES ====================

@app.route("/")
def home():
    """Home page"""
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("login_enhanced.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration"""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        phone = request.form.get("phone", "")
        email = request.form.get("email", "")
        farm_location = request.form.get("farm_location", "")
        
        try:
            conn = sqlite3.connect("agriculture_database.db")
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, phone, email, farm_location) VALUES (?, ?, ?, ?, ?)",
                     (username, password, phone, email, farm_location))
            conn.commit()
            conn.close()
            
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("home"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose another.", "error")
            return redirect(url_for("register"))
    
    return render_template("register_enhanced.html")


@app.route("/login", methods=["POST"])
def login():
    """User login"""
    username = request.form["username"]
    password = request.form["password"]
    
    conn = sqlite3.connect("agriculture_database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    
    if user:
        session["username"] = username
        session["user_id"] = user[0]
        flash(f"Welcome back, {username}!", "success")
        return redirect(url_for("dashboard"))
    else:
        flash("Invalid credentials. Please try again.", "error")
        return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    """Main dashboard"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    # Get user stats
    conn = sqlite3.connect("agriculture_database.db")
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM crop_predictions WHERE username=?", (session["username"],))
    crop_predictions_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM yield_predictions WHERE username=?", (session["username"],))
    yield_predictions_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM disease_predictions WHERE username=?", (session["username"],))
    disease_predictions_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM irrigation_recommendations WHERE username=?", (session["username"],))
    irrigation_count = c.fetchone()[0]
    
    conn.close()
    
    stats = {
        'crop_predictions': crop_predictions_count,
        'yield_predictions': yield_predictions_count,
        'disease_checks': disease_predictions_count,
        'irrigation_recommendations': irrigation_count
    }
    
    return render_template("dashboard.html", username=session["username"], stats=stats)


@app.route("/crop_recommendation", methods=["GET", "POST"])
def crop_recommendation():
    """Crop recommendation page"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        if crop_recommender is None:
            flash("Crop recommendation model not loaded. Please train models first.", "error")
            return redirect(url_for("crop_recommendation"))
        
        try:
            # Get form data
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            
            # Make prediction
            recommended_crop, confidence = crop_recommender.recommend_crop(
                N, P, K, temperature, humidity, ph, rainfall
            )
            
            # Save to database
            conn = sqlite3.connect("agriculture_database.db")
            c = conn.cursor()
            c.execute("""INSERT INTO crop_predictions 
                        (username, N, P, K, temperature, humidity, ph, rainfall, recommended_crop, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (session["username"], N, P, K, temperature, humidity, ph, rainfall,
                      recommended_crop, confidence))
            conn.commit()
            conn.close()
            
            return render_template("crop_result.html", 
                                 crop=recommended_crop, 
                                 confidence=confidence*100 if confidence else None,
                                 params={'N': N, 'P': P, 'K': K, 'temp': temperature,
                                        'humidity': humidity, 'ph': ph, 'rainfall': rainfall})
        
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("crop_recommendation"))
    
    return render_template("crop_recommendation.html")


@app.route("/yield_prediction", methods=["GET", "POST"])
def yield_prediction():
    """Yield prediction page"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        if yield_predictor is None:
            flash("Yield prediction model not loaded. Please train models first.", "error")
            return redirect(url_for("yield_prediction"))
        
        try:
            crop = request.form["crop"]
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            
            # Prepare features (7 base features to match training)
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # Make prediction
            predicted_yield = yield_predictor.predict(features)[0]
            
            # Save to database
            conn = sqlite3.connect("agriculture_database.db")
            c = conn.cursor()
            c.execute("""INSERT INTO yield_predictions 
                        (username, crop, N, P, K, temperature, humidity, ph, rainfall, predicted_yield)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (session["username"], crop, N, P, K, temperature, humidity, ph, rainfall,
                      float(predicted_yield)))
            conn.commit()
            conn.close()
            
            return render_template("yield_result.html", 
                                 crop=crop,
                                 yield_value=round(predicted_yield, 2),
                                 params={'N': N, 'P': P, 'K': K, 'temp': temperature,
                                        'humidity': humidity, 'ph': ph, 'rainfall': rainfall})
        
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("yield_prediction"))
    
    return render_template("yield_prediction.html")


@app.route("/disease_detection", methods=["GET", "POST"])
def disease_detection():
    """Disease risk detection page"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        if disease_detector is None:
            flash("Disease detection model not loaded. Please train models first.", "error")
            return redirect(url_for("disease_detection"))
        
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            
            # Make prediction
            risk_level, confidence = disease_detector.predict_disease_risk(
                N, P, K, temperature, humidity, ph, rainfall
            )
            
            # Get recommendations
            recommendations = disease_detector.get_risk_recommendations(risk_level)
            
            # Save to database
            conn = sqlite3.connect("agriculture_database.db")
            c = conn.cursor()
            c.execute("""INSERT INTO disease_predictions 
                        (username, N, P, K, temperature, humidity, ph, rainfall, risk_level, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (session["username"], N, P, K, temperature, humidity, ph, rainfall,
                      risk_level, confidence))
            conn.commit()
            conn.close()
            
            return render_template("disease_result.html", 
                                 risk_level=risk_level,
                                 confidence=confidence*100 if confidence else None,
                                 recommendations=recommendations,
                                 params={'N': N, 'P': P, 'K': K, 'temp': temperature,
                                        'humidity': humidity, 'ph': ph, 'rainfall': rainfall})
        
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("disease_detection"))
    
    return render_template("disease_detection.html")


@app.route("/irrigation_recommendation", methods=["GET", "POST"])
def irrigation_recommendation():
    """Irrigation recommendation page"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        if irrigation_system is None:
            flash("Irrigation system not initialized.", "error")
            return redirect(url_for("irrigation_recommendation"))
        
        try:
            crop = request.form["crop"]
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            soil_moisture = float(request.form["soil_moisture"])
            rainfall_7days = float(request.form["rainfall_7days"])
            growth_stage = request.form.get("growth_stage", "vegetative")
            soil_type = request.form.get("soil_type", "loam")
            
            # Get recommendation
            recommendation = irrigation_system.recommend_irrigation(
                crop, temperature, humidity, soil_moisture, rainfall_7days,
                growth_stage=growth_stage, soil_type=soil_type
            )
            
            # Get tips
            tips = irrigation_system.get_irrigation_tips(crop, 'normal')
            
            # Save to database
            conn = sqlite3.connect("agriculture_database.db")
            c = conn.cursor()
            c.execute("""INSERT INTO irrigation_recommendations 
                        (username, crop, soil_moisture, temperature, humidity, rainfall_7days,
                         irrigation_needed, amount_mm, urgency)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (session["username"], crop, soil_moisture, temperature, humidity, rainfall_7days,
                      recommendation['irrigation_needed'], recommendation['amount_mm'], 
                      recommendation['urgency']))
            conn.commit()
            conn.close()
            
            return render_template("irrigation_result.html", 
                                 crop=crop,
                                 recommendation=recommendation,
                                 tips=tips)
        
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("irrigation_recommendation"))
    
    return render_template("irrigation_recommendation.html")


@app.route("/history")
def history():
    """View prediction history"""
    if "username" not in session:
        return redirect(url_for("home"))
    
    conn = sqlite3.connect("agriculture_database.db")
    c = conn.cursor()
    
    # Get recent predictions
    c.execute("SELECT * FROM crop_predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10",
             (session["username"],))
    crop_history = c.fetchall()
    
    c.execute("SELECT * FROM yield_predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10",
             (session["username"],))
    yield_history = c.fetchall()
    
    c.execute("SELECT * FROM disease_predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10",
             (session["username"],))
    disease_history = c.fetchall()
    
    c.execute("SELECT * FROM irrigation_recommendations WHERE username=? ORDER BY timestamp DESC LIMIT 10",
             (session["username"],))
    irrigation_history = c.fetchall()
    
    conn.close()
    
    return render_template("history.html",
                         crop_history=crop_history,
                         yield_history=yield_history,
                         disease_history=disease_history,
                         irrigation_history=irrigation_history)


@app.route("/logout")
def logout():
    """User logout"""
    session.pop("username", None)
    session.pop("user_id", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))


# API endpoints for mobile/external integration
@app.route("/api/crop_recommend", methods=["POST"])
def api_crop_recommend():
    """API endpoint for crop recommendation"""
    try:
        data = request.get_json()
        crop, confidence = crop_recommender.recommend_crop(
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        )
        return jsonify({
            'status': 'success',
            'crop': crop,
            'confidence': float(confidence) if confidence else None
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Load ML models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n" + "="*80)
        print(" MODELS NOT LOADED")
        print("="*80)
        print("To train models, run: python train_models.py")
        print("The web app will still run, but predictions will not work.")
        print("="*80 + "\n")
    
    # Run app
    print("\n" + "="*80)
    print(" SMART AGRICULTURE SYSTEM - WEB APPLICATION")
    print("="*80)
    print("Starting Flask development server...")
    print("Access the app at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
