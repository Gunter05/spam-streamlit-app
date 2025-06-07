import subprocess

# Installer les bibliothèques depuis requirements.txt
subprocess.run(["pip", "install", "-r", "requirements.txt"])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Interface

# configuration de la page
st.set_page_config(page_title="Détection de SPAM par IA", page_icon="📨", layout="centered")

# en-tête
st.title("📨 AI-powered SPAM detector")
st.markdown("This system uses a Machine Learning model to predict whether a message is SPAM or not.")

# zone de saisie
message = st.text_area("✍ Write an SMS or email to analyze. :", height=150)

# prediction via api
api_url = "https://spam-api-q58t.onrender.com/predict"

if st.button("🔍 Analyze the message"):
    if message.strip() == "":
        st.warning("Thank you for writing a message to analyze.")
    else:
        try:
            response = requests.post(api_url, json={"message": message})
            if response.status_code == 200:
                result = response.json()
                st.write("✅ Résultat :", result["prediction"])
                st.write("📊 Probabilités :")
                st.progress(int(result["probabilities"]["SPAM"] * 100))
                st.write(f"SPAM : {result['probabilities']['SPAM'] * 100:.2f}%")
                st.write(f"HAM : {result['probabilities']['HAM'] * 100:.2f}%")
            else:
                st.error(f"API error : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Unable to contact the API. Detail : {e}")

# import de fichier
st.markdown("---")
st.subheader("📂 Analyze messages from a .txt file")

uploaded_file = st.file_uploader("Upload a text file (.txt) containing messages, one per line :", type=["txt"])

if uploaded_file is not None:
    #lire les lignes du fichier
    lines = uploaded_file.read().decode("utf-8").splitlines()
    messages = [line.strip() for line in lines if line.strip() != ""]

    if not messages:
        st.warning("The file appears to be empty or incorrectly formatted.")
    else:
        #nettoyage + vectorisation
        cleaned_messages = [clean_text(msg) for msg in messages]
        vect_messages = vectorizer.transform(cleaned_messages)
        predictions = model.predict(vect_messages)
        #resultats dans un tableau
        results_df = pd.DataFrame({
            "Message": messages,
            "Résultat": ["SPAM" if pred == 1 else "HAM" for pred in predictions]
        })
        st.success(f"{len(messages)} Analyzed messages..")
        st.dataframe(results_df, use_container_width=True)
        #visualisation du nombre de SPAM vs HAM
        st.markdown("### Distribution of classified messages")
        count_spam = (results_df["Result"] == "SPAM").sum()
        count_ham = (results_df["Result"] == "HAM").sum()
        fig, ax = plt.subplots()
        ax.pie([count_ham, count_spam], labels=["HAM", "SPAM"], colors=["green", "red"], autopct='%1.1f%%')
        ax.set_title("Distribution of predictions.")
        st.pyplot(fig)
        #telecharger les resultats ?
        st.download_button(
            label="📥 Download the results (CSV)",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="results_spam.csv",
            mime="text/csv"
        )
