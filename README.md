# AutoJudge: Programming Problem Difficulty Predictor

> An intelligent machine learning system that automatically predicts the difficulty of competitive programming problems based on textual descriptions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Model Accuracy](https://img.shields.io/badge/accuracy-52.37%25-brightgreen.svg)
![Dataset](https://img.shields.io/badge/dataset-4112%20problems-orange.svg)

### DEMO VIDEO LINK
 🔗 https://drive.google.com/file/d/1QnajMguOVEdRzaNxtkJ8TzxwOxOb_NdL/view?usp=sharing

## 🎯 Overview

AutoJudge is a comprehensive machine learning solution that predicts competitive programming problem difficulty from problem descriptions. The system performs **dual-task learning**:

1. **Classification**: Predict difficulty class (Easy, Medium, Hard) with **52.37% accuracy**
2. **Regression**: Predict numerical difficulty score (1-10) with **MAE = 1.67**

The project includes a fully functional Streamlit web interface for interactive predictions with confidence scores and probability distributions.

## ✨ Key Features

- **Dual-Task Learning**: Classification + Regression models for comprehensive difficulty prediction
- **Advanced Feature Engineering**: 15 hand-crafted features + 100 TF-IDF vectorization features (115 total)
- **Multiple Models**: Tested 6 different models and selected best performers
- **Web Interface**: Interactive Streamlit application for real-time predictions
- **Production-Ready**: Serialized models, scalers, and vectorizers included
- **Confidence Metrics**: Probability distributions for all predictions
- **Comprehensive Documentation**: Complete project report and setup guide

## 📊 Dataset

- **Source**: Kattis Online Judge (kattis.com)
- **Size**: 4,112 competitive programming problems
- **Distribution**:
  - Easy: 766 problems (18.6%)
  - Medium: 1,405 problems (34.2%)
  - Hard: 1,941 problems (47.2%)
- **Score Range**: 1.1 to 9.7
- **Average Score**: 5.11 ± 2.18

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- 2 GB RAM minimum
- 500 MB disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/autojudge.git
cd autojudge
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

**Start the Streamlit web interface**:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Input a problem** and click "Predict Difficulty" to see:
- Predicted difficulty class (Easy/Medium/Hard)
- Predicted difficulty score (1-10)
- Confidence percentage
- Probability distribution for each class

## 🏗️ Project Structure

```
AutoJudge/
├── README.md
├── requirements.txt
├── app.py
├── problems_data.jsonl
├── autojudge_models.pkl
├── AutoJudge.ipynb
├── AutoJudge_Project_Report.pdf
└── demo_video_link


```

## 🔬 Technical Approach

### Feature Engineering (115 total features)

#### Engineered Features (15 features)

**Text Metrics (3)**:
- Text length (10-7,582 characters)
- Word count (1-1,226 words)
- Average word length (4.66-10.42 characters)

**Mathematical & Algorithm Keywords (7)**:
- Math symbols count (±, ∑, √, ∞, ≤, ≥, etc.)
- Easy keywords frequency ('sum', 'count', 'find', 'simple', 'max', 'min')
- Medium keywords frequency ('graph', 'tree', 'sort', 'algorithm', 'dynamic')
- Hard keywords frequency ('suffix', 'flow', 'convex', 'tarjan', 'kmp', 'trie')
- Bracket count (parentheses, brackets, braces)
- Number count (numeric digits)

**Structural Features (5)**:
- Unique characters (character diversity)
- Constraint count ('constraint', 'limit', 'maximum', 'minimum')
- Algorithm keywords ('dijkstra', 'bfs', 'dfs', 'binary', 'dp', 'greedy')
- Math keywords ('prove', 'theorem', 'formula', 'matrix', 'vector')
- Sentence metrics (count and average length)

#### TF-IDF Vectorization (100 features)

- Configuration: Max 100 features, (1,2)-grams, min_df=5, max_df=0.8
- Captures semantic content with ngrams
- English stop words removed

### Models Used

#### Classification (3 models tested)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Logistic Regression** | **0.5237** | ✓ **Selected** - Best performance |
| Random Forest | 0.5115 | 100 trees, max_depth=20 |
| XGBoost | 0.4994 | 100 estimators, max_depth=7 |

#### Regression (3 models tested)

| Model | R² Score | MAE | RMSE | Notes |
|-------|----------|-----|------|-------|
| Random Forest | 0.1457 | 1.7004 | 2.0361 | 100 trees |
| Gradient Boosting | 0.1465 | 1.6876 | 2.0351 | 100 estimators |
| **XGBoost** | **0.1512** | **1.6711** | **2.0295** | ✓ **Selected** - Best R² |

### Data Preprocessing

1. **Text Combination**: Merge title, description, input_spec, output_spec
2. **Normalization**: Lowercase conversion, standardize class names
3. **Splitting**: Stratified 80:20 train-test split (3,289 train, 823 test)
4. **Scaling**: StandardScaler on engineered features (TF-IDF self-normalized)

## 📈 Results

### Classification Performance

**Overall Accuracy**: 52.37% (Logistic Regression)

**Confusion Matrix**:
```
              Predicted
           Easy  Hard  Medium
Actual Easy   58    49     46
      Hard    28   300     61
     Medium   26   182     73
```

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Easy | 0.52 | 0.38 | 0.44 | 153 |
| Hard | 0.56 | 0.77 | 0.65 | 389 |
| Medium | 0.41 | 0.26 | 0.32 | 281 |
| **Weighted Avg** | **0.50** | **0.52** | **0.50** | **823** |

**Key Insights**:
- Strong at identifying Hard problems (77% recall)
- Struggles with Medium class (25% recall) - boundary class issue
- 57% relative improvement over random baseline (33.3%)

### Regression Performance

**Best Model**: XGBoost Regressor
- **MAE**: 1.6711 (predictions within ±1.67 points on average)
- **RMSE**: 2.0295 (penalizes larger errors)
- **R² Score**: 0.1512 (explains 15.12% of variance)
- **68% Confidence**: Predictions within ±1.67 with 68% probability

**Error Analysis**:
- 1σ Error Range: ±1.67 points
- 2σ Error Range: ±3.34 points
- Score range: 1.1-9.7 (8.6 point span)

### Sample Predictions

| Problem Title | Predicted Class | Score | Confidence |
|---------------|-----------------|-------|-----------|
| Sum of Array | Easy | 1.56 | 70.54% |
| Shortest Path | Easy | 4.36 | 39.50% |
| Maximum Flow Network | Hard | 4.47 | 56.43% |
| Convex Hull Geometry | Hard | 3.93 | 52.95% |

## 💡 Usage Examples

### Running Predictions Programmatically

```python
import pickle
import pandas as pd

# Load trained models
with open('autojudge_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Use the predict_difficulty function from notebook
result = predict_difficulty(
    title="Sum of Array",
    description="Given an array of N integers, find the sum",
    input_desc="First line: N. Next N lines: integers",
    output_desc="Print the sum"
)

print(result)
# Output:
# {
#     'predicted_class': 'Easy',
#     'predicted_score': 1.56,
#     'confidence': 70.54,
#     'class_probabilities': {
#         'Easy': 70.54,
#         'Medium': 21.29,
#         'Hard': 8.18
#     }
# }
```

### Via Web Interface

1. Launch Streamlit: `streamlit run app.py`
2. Enter problem details in text fields
3. Click "Predict Difficulty"
4. View predictions and confidence scores

## 📝 Files Description

### `AutoJudge.ipynb`
Complete Jupyter notebook containing:
- Data loading and exploration
- Feature engineering pipeline
- Model training and evaluation
- Prediction testing
- Visualization and analysis

### `app.py`
Streamlit web application with:
- User input interface (title, description, I/O specs)
- Real-time prediction with loaded models
- Confidence score display
- Class probability distribution
- Modern UI styling

### `autojudge_models.pkl`
Serialized model package containing:
- Logistic Regression classifier
- XGBoost regressor
- StandardScaler for feature scaling
- TF-IDF vectorizer
- Label encoder
- Feature column names

### `problems_data.jsonl`
Complete dataset with 4,112 problems:
- JSONL format (one JSON object per line)
- 8.4 MB file size
- Includes all text fields and difficulty labels

### `AutoJudge Project Report.pdf`
Comprehensive 8+ page project report including:
- Problem statement and motivation
- Dataset description and statistics
- Data preprocessing methodology
- Feature engineering details
- Model architecture and selection
- Complete results and evaluation metrics
- Web interface documentation
- Conclusions and future work

## 🔧 Configuration

### Modifying Model Parameters

Edit `AutoJudge.ipynb` to adjust:

**TF-IDF Configuration** (Cell 7):
```python
tfidf = TfidfVectorizer(
    max_features=100,        # Increase for more features
    ngram_range=(1, 2),      # Change to (1,3) for trigrams
    min_df=5,                # Minimum document frequency
    max_df=0.8               # Maximum document frequency
)
```

**Classification Model** (Cell 9):
```python
clf_lr = LogisticRegression(
    max_iter=1000,           # Increase iterations if needed
    random_state=42,
    n_jobs=-1                # Use all cores
)
```

**Regression Model** (Cell 11):
```python
reg_xgb = XGBRegressor(
    n_estimators=100,        # Number of boosting rounds
    learning_rate=0.1,       # Learning rate
    max_depth=7,             # Tree depth
    random_state=42
)
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Install missing dependency
```bash
pip install streamlit
```

### Issue: "FileNotFoundError: autojudge_models.pkl"
**Solution**: Run the notebook first to generate models
```bash
jupyter notebook AutoJudge.ipynb
# Run all cells
```

### Issue: Slow prediction on first run
**Solution**: First prediction includes model loading; subsequent predictions are faster (<1ms)

### Issue: Low accuracy on new domains
**Solution**: Model trained on Kattis problems; may need retraining for other platforms

## 📚 Methodology Notes

### Why Logistic Regression for Classification?

Despite simpler architecture, Logistic Regression outperformed Random Forest and XGBoost due to:
- Good linear separability in 115-dimensional feature space
- Less prone to overfitting than complex ensemble methods
- Faster training and inference
- Interpretable decision boundaries
- Provides well-calibrated probability estimates

### Why Lower R² for Regression?

R² = 0.1512 indicates:
- Problem difficulty scoring is inherently complex
- Score variance influenced by unlabeled factors (time limits, memory constraints)
- Text-only features insufficient for precise scoring
- Non-linear patterns hard to capture from descriptions alone

### Addressing Class Imbalance

Hard problems dominate dataset (47.2%), leading to:
- Biased predictions toward Hard class
- Lower Medium class recall (25.98%)
- Future improvements: SMOTE, class weights, threshold tuning

## 🚀 Future Improvements

### Short-term (1-2 weeks)
- [ ] Implement SMOTE for class imbalance handling
- [ ] Add cross-validation for robust evaluation
- [ ] Hyperparameter grid search optimization
- [ ] Extract time/memory constraints as features

### Medium-term (1 month)
- [ ] Fine-tune BERT/RoBERTa language model
- [ ] Implement ensemble voting classifier
- [ ] Add feature importance visualization
- [ ] Cross-platform evaluation (AtCoder, CodeForces)

### Long-term (3 months)
- [ ] Transfer learning from pre-trained models
- [ ] Integrate with real online judge platforms
- [ ] Build problem recommendation system
- [ ] Active learning for continuous improvement

## 📊 Benchmark Comparison

| Method | Accuracy |
|--------|----------|
| Random Baseline | 33.3% |
| **AutoJudge (Logistic Regression)** | **52.37%** |
| **Improvement** | **+57%** |


## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
streamlit>=1.0.0
joblib>=1.1.0
```

## 🎓 Educational Value

This project demonstrates:
- Complete ML pipeline from data to deployment
- Feature engineering best practices
- Model selection and evaluation
- Handling class imbalance
- Web application deployment
- Production-ready code organization

Perfect for:
- Learning machine learning workflows
- Understanding NLP preprocessing
- Studying multi-task learning
- Web app deployment examples

## 📖 Citation

If you use AutoJudge in your research, please cite:

```bibtex
@software{autojudge2026,
  title={AutoJudge: Programming Problem Difficulty Predictor},
  author={Naveen Singh},
  year={2026},
  url={https://github.com/singhnaveen02/autojudge}
}
```

## ⭐ Acknowledgments

- Dataset from [Kattis Online Judge](https://kattis.com/)
- Built with [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [Streamlit](https://streamlit.io/)
- Thanks to open-source ML community

---

**Last Updated**: January 8, 2026

**Project Status**: ✅ Complete and Production-Ready

