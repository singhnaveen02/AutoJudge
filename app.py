import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AutoJudge - Difficulty Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.card {
    background: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.card h4 {
    color: white !important;
}
textarea, input {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== STYLES ====================
st.markdown("""
<style>
:root {
    --primary: #667eea;
    --secondary: #764ba2;
}
.header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 2.5rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
}
.badge-easy { background: #2ecc71; padding: 6px 16px; border-radius: 20px; color: white; font-weight: bold; }
.badge-medium { background: #f39c12; padding: 6px 16px; border-radius: 20px; color: white; font-weight: bold; }
.badge-hard { background: #e74c3c; padding: 6px 16px; border-radius: 20px; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="header">
    <h1>🎯 AutoJudge</h1>
    <h3>Programming Problem Difficulty Predictor</h3>
    <p>Predict Easy / Medium / Hard with AI</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    with open('C:/Users/singh/Codes/ACM_OPEN_PROJECT/AUTO_JUDGE/autojudge_models.pkl', 'rb') as f:
        return pickle.load(f)

# ==================== FEATURE EXTRACTION ====================
def extract_features(text):
    if pd.isna(text):
        text = ""
    text = str(text).lower()
    text_length = len(text)
    word_count = len(text.split())
    avg_word_length = text_length / word_count if word_count else 0

    math_symbols = ['±','∑','∏','∫','√','∞','≤','≥','≠','∈','∀','∃','×','÷']
    math_count = sum(text.count(s) for s in math_symbols)

    easy_keywords = ['sum','count','find','check','simple','basic','easy','max','min','list']
    medium_keywords = ['graph','tree','sort','search','recursion','dynamic','optimization','algorithm','path','traverse']
    hard_keywords = ['suffix','flow','convex','hull','tarjan','segment','lazy','propagation','kmp','trie','hash']

    easy_score = sum(text.count(k) for k in easy_keywords)
    medium_score = sum(text.count(k) for k in medium_keywords)
    hard_score = sum(text.count(k) for k in hard_keywords)

    bracket_count = sum(text.count(c) for c in "()[]{}")
    numbers = sum(c.isdigit() for c in text)
    unique_chars = len(set(text))

    constraint_keywords = ['constraint','limit','maximum','minimum','range','bound']
    constraint_count = sum(text.count(k) for k in constraint_keywords)

    algo_keywords = ['dijkstra','bfs','dfs','binary','heap','queue','stack','dp','greedy']
    algo_count = sum(text.count(k) for k in algo_keywords)

    math_keywords = ['prove','theorem','proof','formula','equation','matrix','vector']
    math_word_count = sum(text.count(k) for k in math_keywords)

    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / sentence_count if sentence_count else word_count

    return {
        'text_length': text_length,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'math_symbols': math_count,
        'easy_keywords': easy_score,
        'medium_keywords': medium_score,
        'hard_keywords': hard_score,
        'bracket_count': bracket_count,
        'number_count': numbers,
        'unique_chars': unique_chars,
        'constraint_count': constraint_count,
        'algo_count': algo_count,
        'math_word_count': math_word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length
    }

# ==================== PREDICTION ====================
def predict(models, title, description, input_desc, output_desc):
    combined = f"{title} {description} {input_desc} {output_desc}"

    features_df = pd.DataFrame([extract_features(combined)])
    tfidf_matrix = models['tfidf'].transform([combined])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

    X_new = pd.concat([features_df, tfidf_df], axis=1)

    if models['classifier_name'] == 'Logistic Regression':
        X_new = models['scaler'].transform(X_new)

    class_pred = models['classifier'].predict(X_new)[0]
    class_prob = models['classifier'].predict_proba(X_new)[0]
    score_pred = np.clip(models['regressor'].predict(X_new)[0], 1, 10)

    class_name = models['label_encoder'].inverse_transform([class_pred])[0]

    return {
        'class': class_name,
        'score': round(score_pred, 2),
        'confidence': float(max(class_prob)) * 100,
        'probs': {
            models['label_encoder'].inverse_transform([i])[0]: float(class_prob[i]) * 100
            for i in range(len(models['label_encoder'].classes_))
        }
    }

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 📊 About AutoJudge")
    st.info("""
    Predicts difficulty of programming problems using ML & NLP.

    **Outputs**
    - Difficulty Class
    - Difficulty Score (1–10)
    - Confidence

    """)

# ==================== MAIN ====================
models = load_models()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Feature Info", "ℹ️ Help"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h4>Problem Details</h4>", unsafe_allow_html=True)
        title = st.text_input("Problem Title")
        description = st.text_area("Problem Description", height=140)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h4>Input / Output</h4>", unsafe_allow_html=True)
        input_desc = st.text_area("Input Format", height=120)
        output_desc = st.text_area("Output Format", height=120)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Predict Difficulty", type="primary", use_container_width=True):
        if not title or not description:
            st.warning("Please provide title and description.")
        else:
            with st.spinner("Analyzing..."):
                result = predict(models, title, description, input_desc, output_desc)

            badge = f"badge-{result['class'].lower()}"

            st.markdown(f"""
            <div class="result-card">
                <h2>Predicted Difficulty</h2>
                <h1><span class="{badge}">{result['class'].upper()}</span></h1>
                <p>Confidence: {result['confidence']:.1f}%</p>
                <h3>Score: {result['score']} / 10</h3>
            </div>
            """, unsafe_allow_html=True)

            prob_df = pd.DataFrame(result['probs'].items(), columns=["Class", "Probability"]).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Class"))

with tab2:
    st.markdown("### 📈 Feature Info")
    st.write("""
    - Text length, word count, sentence count  
    - Keyword indicators for easy / medium / hard  
    - Math, graph, algorithm hints  
    - TF-IDF vectorization of problem text  
    """)

with tab3:
    st.markdown("### ℹ️ Help")
    st.write("""
    1. Enter full problem description.
    2. Include constraints and formats.
    3. Click Predict.
    4. View difficulty and score.
    """)

st.markdown("---")
st.markdown("<center>🚀 AutoJudge</center>", unsafe_allow_html=True)
