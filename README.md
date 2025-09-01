# UPAIR: Understandable Post-hoc AI Reports

## Abstract

Clinical adoption of machine learning (ML) in medical imaging is limited by the lack of interpretability. To address this, we present UPAIR (Understandable Post-hoc AI Reports), a pipeline designed to generate transparent, evidence-based explanations by combining SHAP analysis with retrieval-augmented generation (RAG) and large language models (LLMs). We trained 12 Classifiers to predict IDH mutation status in glioma using radiomics and clinical features. SHAP values were used to identify key contributors to each prediction. Relevant literature was retrieved from a curated PubMed dataset using FAISS for similarity search and passed through a RAG framework with Gemini 2.5 Pro to generate concise, reference-supported explanations for each feature. The model achieved a best AUROC of 86.7% on a hold-out test set using Gradient Boosting Classifier. In a case study of a single patient excluded from training, the model predicted IDH-wildtype glioma. SHAP identified MGMT status, age, and three radiomic features as the most influential. UPAIR produced a structured report combining SHAP visualizations with LLM-generated summaries grounded in scientific evidence. UPAIR provides a practical, model-agnostic framework that enhances ML interpretability in clinical settings, helping bridge the gap between black-box AI and real-world medical decision-making.

---

## 🚀 Pipeline Overview

This project implements an end-to-end pipeline to:
1.  **Train ML Models**: Trains and evaluates multiple classifiers to predict IDH mutation status from radiomics and clinical data.
2.  **Explain Predictions**: Uses SHAP (SHapley Additive exPlanations) to identify the key features driving a specific prediction for a single patient.
3.  **Retrieve Evidence**: Builds a vector database (FAISS) from a corpus of medical literature and retrieves articles relevant to the key predictive features.
4.  **Generate Reports**: Combines the prediction, SHAP analysis, and retrieved literature into a clear, evidence-based clinical report using a Large Language Model (Gemini 2.5 Pro).

---

## 🔧 Setup and Installation

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd UPAIR_Project
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
You may also need to install system-level dependencies for `weasyprint` and `pdf2image`:
```bash
# For Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y libpangocairo-1.0-0 poppler-utils

# For Fedora/CentOS
sudo yum install -y pango poppler-utils
```

**4. Add Your Data**
- Place your feature CSV file (e.g., `UCSF+UPENN_all_features.csv`) inside the `data/` directory.
- If you need to extract radiomics features, place the raw NIfTI files in a structured way inside the `data/` directory.

**5. Configure the Pipeline**
- Open `src/config.py` to set file paths, API keys, and other parameters for your environment.

---

## ▶️ How to Run

Execute the main pipeline script from the root directory of the project:

```bash
python src/main.py
```

The script will perform all the steps—from data loading and model training to generating the final PDF report—and save the output in the `reports/` folder.

---

## 📂 Project Structure

-   `data/`: Directory for input data files.
-   `reports/`: Directory where output reports (images, PDFs) are saved.
-   `src/`: Contains the core Python source code.
    -   `config.py`: Central configuration for file paths, model parameters, and API keys.
    -   `data_handler.py`: Handles data loading and preprocessing.
    -   `model_trainer.py`: Manages the training and evaluation of ML models.
    -   `explainer.py`: Generates SHAP explanations for model predictions.
    -   `retriever.py`: Builds the FAISS index and retrieves relevant literature.
    -   `report_generator.py`: Uses an LLM to generate text and compiles the final PDF report.
    -   `main.py`: The main script that orchestrates the entire pipeline.
-   `requirements.txt`: A list of all Python packages required for the project.
