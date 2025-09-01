import config
from data_handler import DataHandler
from model_trainer import ModelTrainer
from explainer import Explainer
from retriever import FaissRetriever, get_paper_texts
from report_generator import ReportGenerator

def main():
    """
    Orchestrates the entire UPAIR pipeline.
    """
    print("--- Starting UPAIR Pipeline ---")

    # 1. Load and Prepare Data
    data_handler = DataHandler()
    if not data_handler.load_data():
        return # Exit if data loading fails
    data_handler.prepare_and_split_data()

    # 2. Train Model and Find the Best One
    trainer = ModelTrainer(config.CLASSIFIERS)
    trainer.train_and_evaluate(
        data_handler.X_train, data_handler.y_train,
        data_handler.X_test, data_handler.y_test
    )

    # 3. Make Prediction on a Single Sample
    raw_prediction, predicted_class = trainer.predict_sample(data_handler.sample_X)
    prediction_info = {
        'model_name': trainer.best_classifier_name,
        'predicted_class': predicted_class,
        'raw_prediction': raw_prediction
    }

    # 4. Explain the Prediction with SHAP
    explainer = Explainer(trainer.best_model, data_handler.X_train)
    explainer.fit_explainer()
    top_features, shap_plot_path = explainer.explain_sample(data_handler.sample_X)
    
    if top_features is None:
        print("❌ Could not generate SHAP explanations. Exiting.")
        return

    # 5. Build Retriever and Get Context
    # This step simulates downloading and processing papers.
    paper_texts = get_paper_texts()
    faiss_retriever = FaissRetriever()
    faiss_retriever.build_index_from_texts(paper_texts)
    retrieved_context = faiss_retriever.retrieve_context(top_features)

    # 6. Generate the Final Report
    report_generator = ReportGenerator(api_key=config.GEMINI_API_KEY)
    llm_explanation = report_generator.generate_llm_explanation(
        predicted_class, top_features, retrieved_context
    )
    
    report_generator.create_pdf_report(
        sample_info=data_handler.sample_info,
        prediction_info=prediction_info,
        shap_plot_path=shap_plot_path,
        llm_text=llm_explanation
    )

    print("\n--- UPAIR Pipeline Finished ---")

if __name__ == '__main__':
    main()