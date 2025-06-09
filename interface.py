import subprocess

# Installer les bibliothèques depuis requirements.txt
subprocess.run(["pip", "install", "-r", "requirements.txt"])

import streamlit as st
import string
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Chargement des stopwords
stop_words = set(stopwords.words("english"))

# Fonction de nettoyage
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Interface

# configuration de la page
st.set_page_config(page_title="Détection de SPAM par IA", page_icon="📨", layout="centered")
st.title("📨 AI-powered SPAM detector")
st.markdown("This system uses a Machine Learning model to predict whether a message is SPAM or not.")

# zone de saisie
message = st.text_area("✍ Write an SMS to analyze:", height=150)

# prediction via api
api_url = "http://127.0.0.1:5000/predict"

if st.button("🔍 Analyze the message"):
    if message.strip() == "":
        st.warning("Please write a message.")
    else:
        try:
            response = requests.post(api_url, json={"message": message})
            if response.status_code == 200:
                result = response.json()
                st.write("✅ Result :", result["prediction"])
                st.write("📊 Probabilities :")
                st.progress(int(result["probabilities"]["SPAM"] * 100))
                st.write(f"SPAM : {result['probabilities']['SPAM'] * 100:.2f}%")
                st.write(f"HAM : {result['probabilities']['HAM'] * 100:.2f}%")
            else:
                st.error(f"API error : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Could not contact API. Detail : {e}")

# --------------------------
# PARTIE 2 : fichier .txt
# --------------------------
st.markdown("---")
st.subheader("📂 Analyze messages from a .txt file")

uploaded_file = st.file_uploader("Upload a .txt file with one message per line:", type=["txt"])

def basic_clean(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    messages = [line.strip() for line in lines if line.strip() != ""]

    if not messages:
        st.warning("The file appears to be empty or incorrectly formatted.")
    else:
        predictions = []

        with st.spinner("Analyzing messages..."):
            for msg in messages:
                try:
                    response = requests.post(api_url, json={"message": msg})
                    if response.status_code == 200:
                        result = response.json()
                        predictions.append(result["prediction"])
                    else:
                        predictions.append("Erreur")
                except Exception as e:
                    predictions.append("Erreur")

        # Affichage des résultats
        results_df = pd.DataFrame({
            "Message": messages,
            "Prediction": predictions
        })

        st.success(f"{len(messages)} messages analyzed.")
        st.dataframe(results_df, use_container_width=True)

        # Graphe circulaire
        count_spam = results_df["Prediction"].value_counts().get("SPAM", 0)
        count_ham = results_df["Prediction"].value_counts().get("HAM", 0)

        fig, ax = plt.subplots()
        ax.pie([count_ham, count_spam], labels=["HAM", "SPAM"],
               colors=["green", "red"], autopct='%1.1f%%')
        ax.set_title("Spam vs Ham Messages")
        st.pyplot(fig)

        # Téléchargement CSV
        st.download_button(
            label="📥 Download the results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="spam_results.csv",
            mime="text/csv"
        )

# --------------------------
# PARTIE 3 : historique 
# --------------------------
st.markdown("---")
st.subheader("🧾 Message History")

filter_option = st.selectbox("Filter by prediction:", ["All", "SPAM only", "HAM only"])

# Appliquer le filtre
if filter_option == "SPAM only":
    filtered = [m for m in st.session_state.history if m["Prediction"] == "SPAM"]
elif filter_option == "HAM only":
    filtered = [m for m in st.session_state.history if m["Prediction"] == "HAM"]
else:
    filtered = st.session_state.history

# Affichage sous forme de tableau
if filtered:
    df_history = pd.DataFrame(filtered)
    st.dataframe(df_history, use_container_width=True)
    st.download_button("📥 Download history (CSV)", df_history.to_csv(index=False).encode("utf-8"),
                       "history_spam.csv", mime="text/csv")
else:
    st.info("No messages to display yet.")
