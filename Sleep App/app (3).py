from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import joblib
import pandas as pd
import hashlib
from functools import wraps

app = Flask(__name__)
app.secret_key = "your_secret_key_12345"

# -----------------------------
# MySQL / XAMPP database config
# -----------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "sleep_app_db"
}

# -----------------------------
# FullPipeline class
# -----------------------------
class FullPipeline:
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.classes_ = classifier.classes_

    def _add_features(self, df):
        df = df.copy()

        df["sleep_risk"] = (
            (10 - df["Quality of Sleep"]) +
            df["Stress Level"] +
            (8 - df["Sleep Duration"].clip(0, 8))
        )
        df["bp_risk"] = df["Systolic"] + df["Diastolic"]

        df["activity_score"] = (
            (df["Physical Activity Level"] / 10.0) +
            (df["Daily Steps"] / 1000.0)
        ) / 2.0

        df["health_index"] = (
            df["Quality of Sleep"] +
            (10 - df["Stress Level"]) +
            (df["Sleep Duration"].clip(0, 10))
        ) / 3.0

        return df

    def predict(self, X):
        X = self._add_features(X)
        return self.classifier.predict(self.preprocessor.transform(X))

    def predict_proba(self, X):
        X = self._add_features(X)
        return self.classifier.predict_proba(self.preprocessor.transform(X))


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("sleep_disorder_pipeline.pkl")


# -----------------------------
# Database connection
# -----------------------------
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


# -----------------------------
# Password hash
# -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# -----------------------------
# Login required decorator
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in first.", "user_error")
            return redirect(url_for("signin"))
        return f(*args, **kwargs)
    return decorated_function


# -----------------------------
# Admin login required decorator
# -----------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session:
            flash("Admin access required.", "admin_error")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Signup
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not full_name or not email or not password or not confirm_password:
            flash("All fields are required.", "user_error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.", "user_error")
            return redirect(url_for("signup"))

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                cursor.close()
                conn.close()
                flash("Email already registered.", "user_error")
                return redirect(url_for("signup"))

            hashed_pw = hash_password(password)

            cursor.execute("""
                INSERT INTO users (full_name, email, password, created_at)
                VALUES (%s, %s, %s, NOW())
            """, (full_name, email, hashed_pw))

            conn.commit()
            cursor.close()
            conn.close()

            flash("Registration successful. Please sign in.", "user_success")
            return redirect(url_for("signin"))

        except Exception as e:
            print("Signup error:", e)
            flash(f"Signup error: {str(e)}", "user_error")
            return redirect(url_for("signup"))

    return render_template("signup.html")


# -----------------------------
# Signin
# -----------------------------
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_pw = hash_password(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT * FROM users
                WHERE email = %s AND password = %s
            """, (email, hashed_pw))

            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session["user_id"] = user["id"]
                session["user_name"] = user["full_name"]
                flash("Login successful.", "user_success")
                return redirect(url_for("predict"))
            else:
                flash("Invalid email or password.", "user_error")
                return redirect(url_for("signin"))

        except Exception as e:
            print("Signin error:", e)
            flash(f"Signin error: {str(e)}", "user_error")
            return redirect(url_for("signin"))

    return render_template("signin.html")


# -----------------------------
# Logout
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    flash("Logged out successfully.", "user_success")
    return redirect(url_for("index"))


# -----------------------------
# Predict
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None
    conn = None
    cursor = None
    user_name = session.get("user_name")

    if request.method == "POST":
        try:
            gender = request.form.get("gender")
            age = int(request.form.get("age"))
            occupation = request.form.get("occupation")
            sleep_duration = float(request.form.get("sleep_duration"))
            quality_of_sleep = int(request.form.get("quality_of_sleep"))
            physical_activity_level = int(request.form.get("physical_activity_level"))
            stress_level = int(request.form.get("stress_level"))
            bmi_category = request.form.get("bmi_category")
            heart_rate = int(request.form.get("heart_rate"))
            daily_steps = int(request.form.get("daily_steps"))
            systolic = int(request.form.get("systolic"))
            diastolic = int(request.form.get("diastolic"))

            input_df = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity_level,
                "Stress Level": stress_level,
                "BMI Category": bmi_category,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps,
                "Systolic": systolic,
                "Diastolic": diastolic
            }])

            input_df["sleep_risk"] = (
                (10 - input_df["Quality of Sleep"]) +
                input_df["Stress Level"] +
                (8 - input_df["Sleep Duration"].clip(0, 8))
            )
            input_df["bp_risk"] = input_df["Systolic"] + input_df["Diastolic"]
            input_df["activity_score"] = (
                (input_df["Physical Activity Level"] / 10.0) +
                (input_df["Daily Steps"] / 1000.0)
            ) / 2.0
            input_df["health_index"] = (
                input_df["Quality of Sleep"] +
                (10 - input_df["Stress Level"]) +
                (input_df["Sleep Duration"].clip(0, 10))
            ) / 3.0

            class_labels = list(getattr(model, "classes_", []))
            if not class_labels:
                class_labels = ["No disease", "Insomnia", "Sleep Apnea"]

            final_output = None
            insomnia_p = 0.0
            apnea_p = 0.0
            no_disease_p = 0.0
            THRESHOLD = 0.30

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                prob_dict = {label: float(p) for label, p in zip(class_labels, proba)}

                insomnia_p = prob_dict.get("Insomnia", 0.0)
                apnea_p = prob_dict.get("Sleep Apnea", 0.0)
                no_disease_p = prob_dict.get("No disease", 0.0)

                if "Sleep Apnea" in class_labels and apnea_p >= THRESHOLD:
                    final_output = "Sleep Apnea"
                else:
                    raw_pred = model.predict(input_df)[0]
                    final_output = str(raw_pred).strip("[]'\"").strip()
            else:
                raw_pred = model.predict(input_df)[0]
                final_output = str(raw_pred).strip("[]'\"").strip()

            prediction = final_output

            conn = get_db_connection()
            conn.autocommit = False
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    user_id, gender, age, occupation, sleep_duration, quality_of_sleep,
                    physical_activity_level, stress_level, bmi_category, heart_rate,
                    daily_steps, systolic, diastolic, prediction,
                    insomnia_prob, sleep_apnea_prob, no_disease_prob, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                session["user_id"],
                gender, age, occupation,
                sleep_duration, quality_of_sleep,
                physical_activity_level, stress_level,
                bmi_category, heart_rate, daily_steps,
                systolic, diastolic,
                str(final_output),
                float(insomnia_p),
                float(apnea_p),
                float(no_disease_p)
            ))

            conn.commit()
            flash(f"Prediction saved: {final_output}", "user_success")

        except Exception as e:
            if conn:
                conn.rollback()
            print("Prediction error:", e)
            flash(f"Prediction error: {str(e)}", "user_error")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    return render_template("predict.html", prediction=prediction, user_name=user_name)


# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT created_at, prediction
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (session["user_id"],))
        history_rows = cursor.fetchall()

        cursor.execute("""
            SELECT prediction, COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            GROUP BY prediction
        """, (session["user_id"],))
        chart_rows = cursor.fetchall()

        cursor.close()
        conn.close()

        chart_labels = [row["prediction"] for row in chart_rows]
        chart_values = [row["total"] for row in chart_rows]
        user_name = session.get("user_name")

        return render_template(
            "dashboard.html",
            history=history_rows,
            chart_labels=chart_labels,
            chart_values=chart_values,
            user_name=user_name
        )

    except Exception as e:
        print("Dashboard error:", e)
        flash(f"Dashboard error: {str(e)}", "user_error")
        return redirect(url_for("predict"))


# -----------------------------
# Admin Login
# -----------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_pw = hash_password(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT * FROM admins
                WHERE username = %s AND password = %s
            """, (username, hashed_pw))

            admin = cursor.fetchone()
            cursor.close()
            conn.close()

            if admin:
                session["admin_id"] = admin["id"]
                session["admin_name"] = admin["username"]
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Invalid admin credentials.", "admin_error")
                return redirect(url_for("admin_login"))

        except Exception as e:
            flash(f"Error: {str(e)}", "admin_error")
            return redirect(url_for("admin_login"))

    return render_template("admin_login.html")


# -----------------------------
# Admin Dashboard
# -----------------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT id, full_name, email, created_at
            FROM users
            ORDER BY created_at DESC
        """)
        users = cursor.fetchall()

        cursor.execute("""
            SELECT p.*, u.full_name, u.email
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        """)
        predictions = cursor.fetchall()

        cursor.execute("""
            SELECT prediction, COUNT(*) as total
            FROM predictions
            GROUP BY prediction
        """)
        chart_rows = cursor.fetchall()

        cursor.close()
        conn.close()

        chart_labels = [row["prediction"] for row in chart_rows]
        chart_values = [row["total"] for row in chart_rows]

        return render_template(
            "admin_dashboard.html",
            users=users,
            predictions=predictions,
            chart_labels=chart_labels,
            chart_values=chart_values,
            total_users=len(users),
            total_predictions=len(predictions)
        )

    except Exception as e:
        flash(f"Error: {str(e)}", "admin_error")
        return redirect(url_for("admin_login"))


# -----------------------------
# Admin Logout
# -----------------------------
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_id", None)
    session.pop("admin_name", None)
    flash("Admin logged out successfully.", "admin_error")
    return redirect(url_for("admin_login"))


if __name__ == "__main__":
    app.run(debug=True)