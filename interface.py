import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Chargement des objets
model = pickle.load(open("model_nb.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
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

# en-tête
st.title("📨 Détecteur de SPAM par IA")
st.markdown("Ce système utilise un modèle de Machine Learning pour prédire si un message est un SPAM ou non.")

# zone de saisie
message = st.text_area("✍ Écris un message SMS ou email à analyser :", height=150)

# prediction via api
st.markdown("### 🔌 Analyse via API Flask locale")

api_url = "http://127.0.0.1:5000/predict"

if st.button("🔍 Envoyer à l'API Flask"):
    if message.strip() == "":
        st.warning("Merci d’écrire un message à analyser.")
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
                st.error(f"Erreur de l’API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Impossible de contacter l'API. Détail : {e}")

# bouton de prédiction
if st.button("🔍 Analyser le message"):
    if message.strip() == "":
        st.warning("Merci d’écrire un message à analyser.")
    else:
        cleaned = clean_text(message)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]

        #resultat
        if prediction == 1:
            st.error("🚨 Ce message est probablement un SPAM.")
        else:
            st.success("✅ Ce message est probablement légitime (HAM).")

        #probabilités
        st.markdown("### 📊 Probabilités de prédiction :")
        st.progress(int(proba[1] * 100))
        st.write(f"SPAM : {proba[1]*100:.2f}%")
        st.write(f"HAM : {proba[0]*100:.2f}%")

        #pied de page
        st.markdown("---")
        st.markdown("🔬 Projet de détection de SPAM — Étudiant(e) IA")

# import de fichier
st.markdown("---")
st.subheader("📂 Analyser des messages depuis un fichier .txt")

uploaded_file = st.file_uploader("Téléverse un fichier texte (.txt) contenant des messages, un par ligne :", type=["txt"])

if uploaded_file is not None:
    #lire les lignes du fichier
    lines = uploaded_file.read().decode("utf-8").splitlines()
    messages = [line.strip() for line in lines if line.strip() != ""]

    if not messages:
        st.warning("Le fichier semble vide ou mal formaté.")
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
        st.success(f"{len(messages)} messages analysés.")
        st.dataframe(results_df, use_container_width=True)
        #visualisation du nombre de SPAM vs HAM
        st.markdown("### Répartition des messages classés")
        count_spam = (results_df["Résultat"] == "SPAM").sum()
        count_ham = (results_df["Résultat"] == "HAM").sum()
        fig, ax = plt.subplots()
        ax.pie([count_ham, count_spam], labels=["HAM", "SPAM"], colors=["green", "red"], autopct='%1.1f%%')
        ax.set_title("Répartition des prédictions")
        st.pyplot(fig)
        #telecharger les resultats ?
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="resultats_spam.csv",
            mime="text/csv"
        )