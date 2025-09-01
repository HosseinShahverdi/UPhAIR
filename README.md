# UPAIR Project - File Structure and Code Organization

## Project Structure:
```
UPAIR/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── radiomics_extractor.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   └── model_trainer.py
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── shap_analyzer.py
│   │   └── feature_importance.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── paper_retriever.py
│   │   ├── embedding_engine.py
│   │   └── faiss_indexer.py
│   ├── report_generation/
│   │   ├── __init__.py
│   │   ├── llm_explainer.py
│   │   └── report_formatter.py
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py
├── notebooks/
│   └── UPAIR_demo.ipynb
├── data/
│   └── README.md
├── outputs/
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_basic.py
```

## File Contents:

### README.md
```markdown
# UPAIR: Understandable Post-hoc AI Reports

UPAIR is a pipeline designed to generate transparent, evidence-based explanations for machine learning predictions in medical imaging by combining SHAP analysis with retrieval-augmented generation (RAG) and large language models (LLMs).

## Abstract

Clinical adoption of machine learning (ML) in medical imaging is limited by the lack of interpretability. To address this, we present UPAIR (Understandable Post-hoc AI Reports), a pipeline designed to generate transparent, evidence-based explanations by combining SHAP analysis with retrieval-augmented generation (RAG) and large language models (LLMs). We trained 12 Classifiers to predict IDH mutation status in glioma using radiomics and clinical features. SHAP values were used to identify key contributors to each prediction. Relevant literature was retrieved from a curated PubMed dataset using FAISS for similarity search and passed through a RAG framework with Gemini 2.5 Pro to generate concise, reference-supported explanations for each feature. The model achieved a best AUROC of 86.7% on a hold-out test set using Gradient Boosting Classifier. In a case study of a single patient excluded from training, the model predicted IDH-wildtype glioma. SHAP identified MGMT status, age, and three radiomic features as the most influential. UPAIR produced a structured report combining SHAP visualizations with LLM-generated summaries grounded in scientific evidence. UPAIR provides a practical, model-agnostic framework that enhances ML interpretability in clinical settings, helping bridge the gap between black-box AI and real-world medical decision-making.

## Features

- **Radiomics Feature Extraction**: Automated extraction of radiomic features from medical images
- **Multi-Classifier Ensemble**: Support for 12 different ML classifiers with hyperparameter optimization
- **SHAP-based Interpretability**: Generate feature importance explanations using SHAP values
- **Literature Retrieval**: Automated retrieval of relevant scientific papers from PubMed
- **RAG Framework**: Context-aware explanation generation using retrieved literature
- **Clinical Report Generation**: Automated generation of clinician-readable reports

## Installation

```bash
git clone https://github.com/yourusername/UPAIR.git
cd UPAIR
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.model_trainer import ModelTrainer
from src.interpretability.shap_analyzer import SHAPAnalyzer
from src.report_generation.llm_explainer import LLMExplainer

# 1. Train model
trainer = ModelTrainer()
model = trainer.train_best_model(X_train, y_train)

# 2. Generate SHAP explanations
analyzer = SHAPAnalyzer(model)
shap_values = analyzer.explain_prediction(sample_X)

# 3. Generate clinical report
explainer = LLMExplainer()
report = explainer.generate_clinical_report(shap_values, sample_X)
```

## Usage

### 1. Feature Extraction
```python
from src.data_processing.radiomics_extractor import RadiomicsExtractor

extractor = RadiomicsExtractor()
features = extractor.extract_features(image_path, mask_path)
```

### 2. Model Training
```python
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.compare_classifiers(X, y)
best_model = trainer.get_best_model()
```

### 3. Generate Explanations
```python
from src.interpretability.shap_analyzer import SHAPAnalyzer
from src.report_generation.llm_explainer import LLMExplainer

# SHAP analysis
analyzer = SHAPAnalyzer(best_model)
shap_values = analyzer.explain_prediction(sample_data)

# Generate report
explainer = LLMExplainer()
report = explainer.generate_report(shap_values, features)
```

## Requirements

- Python 3.8+
- See `requirements.txt` for complete dependencies

## Project Structure

- `src/`: Main source code
- `notebooks/`: Jupyter notebooks for demos and examples
- `data/`: Data directory (add your datasets here)
- `outputs/`: Generated reports and visualizations
- `tests/`: Unit tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use UPAIR in your research, please cite:

```bibtex
@article{upair2024,
  title={UPAIR: Understandable Post-hoc AI Reports for Medical Imaging},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com]
```

### requirements.txt
```
pyradiomics>=3.0.1
nibabel>=3.2.2
pandas>=1.5.3
PyMuPDF>=1.21.1
pdf2image>=3.1.0
beautifulsoup4>=4.11.2
weasyprint>=58.1
sentence-transformers>=2.2.2
faiss-cpu>=1.7.3
requests>=2.28.2
feedparser>=6.0.10
scikit-learn>=1.2.2
lightgbm>=3.3.5
xgboost>=1.7.4
numpy>=1.24.3
matplotlib>=3.7.1
seaborn>=0.12.2
shap>=0.41.0
imbalanced-learn>=0.10.1
scipy>=1.10.1
SimpleITK>=2.2.1
tqdm>=4.65.0
google-generativeai>=0.3.0
pillow>=9.5.0
markdown>=3.4.3
```
