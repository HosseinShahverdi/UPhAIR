import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config

class Explainer:
    """
    Generates SHAP explanations for a model prediction.
    """
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns
        self.shap_explainer = None
        self.preprocessor = None

    def fit_explainer(self):
        """
        Creates a SHAP explainer based on the trained model.
        """
        print("\nInitializing SHAP explainer...")
        
        # Extract the preprocessor steps (imputer + scaler) from the pipeline
        self.preprocessor = self.model[0:2]
        
        # Extract the classifier model
        classifier = self.model.named_steps['clf']
        
        # Transform the training data to be used as the background for the explainer
        X_train_transformed = self.preprocessor.transform(self.X_train)
        
        # SHAP works best with a function, so we wrap the model's predict_proba
        def predict_fn(x):
            return classifier.predict_proba(x)

        self.shap_explainer = shap.KernelExplainer(predict_fn, X_train_transformed)
        print("SHAP explainer fitted.")

    def explain_sample(self, sample_X):
        """
        Calculates SHAP values for a single sample and generates a waterfall plot.
        """
        if not self.shap_explainer:
            print("❌ Error: SHAP explainer not fitted. Call fit_explainer() first.")
            return None, None
        
        print("Explaining prediction for the sample...")
        
        # Transform the single sample using the same preprocessor
        sample_X_transformed = self.preprocessor.transform(sample_X)

        # Calculate SHAP values for the positive class (IDH-wildtype)
        shap_values = self.shap_explainer.shap_values(sample_X_transformed, nsamples=100)[0] # Index 1 for the positive class

        # Generate and save the waterfall plot
        plot_path = os.path.join(config.REPORTS_DIR, config.SHAP_PLOT_FILENAME)
        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=self.shap_explainer.expected_value[1], 
                                             data=sample_X_transformed[0], 
                                             feature_names=self.feature_names),
                           show=False)
        # plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        # plt.close()
        plt.show()
        print(f"SHAP waterfall plot saved to: {plot_path}")

        # Extract top 5 features
        abs_shap = np.abs(shap_values[0])
        top5_idx = np.argsort(abs_shap)[-5:][::-1]

        top_features_formatted = []
        for i in top5_idx:
            feature_name = self.feature_names[i]
            feature_value = sample_X.iloc[0, i]
            shap_value = shap_values[0, i]
            top_features_formatted.append(f"- **{feature_name}**: {feature_value:.4f} (SHAP value: {shap_value:.4f})")

        return top_features_formatted, plot_path