from flask import Flask, render_template, request, jsonify
import joblib
import google.generativeai as genai
import pandas as pd
import numpy as np

import re

def clean_recipe_text(text):
    # Remove **bold markdown**
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    
    # Convert multiple newlines to just one
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Optional: Convert headings to uppercase or highlight them
    text = re.sub(r"(?<=\n)(Recipe|Ingredients|Preparation|Important Considerations.*?):", r"\n\1:", text, flags=re.IGNORECASE)
    
    return text.strip()

# Initialize Flask app
app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load additional data
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

# Reconstruct the LabelEncoder from disease names
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(desc_df["Disease"].unique())

# Configure Gemini API (⚠️ Replace with your actual key)
genai.configure(api_key="YOUR API KEY")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    description = None
    precautions = []
    recipe_text = None

    if request.method == "POST":
        user_input = request.form["symptoms"]
        symptoms = [sym.strip().lower() for sym in user_input.split(",")]
        input_text = " ".join(symptoms)
        input_vector = vectorizer.transform([input_text])
        probabilities = model.predict(input_vector)
        predicted_class_index = np.argmax(probabilities, axis=1)[0]
        predicted_disease = le.inverse_transform([predicted_class_index])[0]
        prediction = predicted_disease  # This is now a clean disease name like "Malaria"

        # Get description
        try:
            description = desc_df.loc[desc_df["Disease"] == predicted_disease, "Description"].values[0]
        except:
            description = "No description available."

        # Get precautions
        try:
            row = precaution_df.loc[precaution_df["Disease"] == predicted_disease]
            precautions = row.values[0][1:].tolist()
        except:
            precautions = ["No precautions available."]

        # Generate recipe
        try:
            gen_model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Suggest a healthy Indian recipe for someone with {predicted_disease}. Include ingredients and preparation steps."
            response = gen_model.generate_content(prompt)
            raw_text = response.text
            recipe_text = clean_recipe_text(raw_text)  # ✅ Clean the response
        except Exception as e:
            recipe_text = f"Error generating recipe: {e}"

    return render_template("index.html", prediction=prediction, description=description,
                           precautions=precautions, recipe_text=recipe_text)

# 🔁 Chatbot API route (POST from chatbox)
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message.strip():
        return jsonify({"response": "Please enter a valid message."})

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_message)
        cleaned_text = clean_recipe_text(response.text)
        return jsonify({"response": cleaned_text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
