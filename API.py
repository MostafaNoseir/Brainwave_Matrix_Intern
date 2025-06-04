import streamlit as st
import joblib

# Load your saved vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('fake_news_model.pkl')

# Streamlit app setup
st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")

st.title("📰 News Topic Classifier")
st.markdown("""
This app detects whether a news article is **True** or **Fake** based on its **title** and **text** content using a machine learning model.
""")

# Input form
with st.form(key='news_form'):
    title = st.text_input("Article Title", placeholder="Enter the news article title here...")
    text = st.text_area("Article Text", height=200, placeholder="Enter the full news article text here...")
    submit_button = st.form_submit_button(label='Classify')

if submit_button:
    if not title.strip() and not text.strip():
        st.error("Please enter at least a title or some text to classify.")
    else:
        # Combine title and text
        combined_text = (title + ' ' + text).strip()

        # Vectorize and predict
        X_vec = vectorizer.transform([combined_text])
        prediction = model.predict(X_vec)[0]

        # If your model uses numeric labels like 0/1, map them here
        label_map = {
            0: '✅ This news is likely TRUE.',
            1: '❌ This news is likely FAKE.'
        }

        # Get friendly label or fallback
        friendly_label = label_map.get(prediction, f"🧐 Classified as: {prediction}")

        st.success("### Prediction Result:")
        st.markdown(f"**{friendly_label}**")

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_vec)[0]
            max_prob = max(probs)
            st.write(f"Confidence: {max_prob:.2%}")

        st.divider()
        
st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; font-size: 15px; color: gray;'>
        Created by Mostafa Noseir
    </div>
    """,
    unsafe_allow_html=True
)