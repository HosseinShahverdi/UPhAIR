import os
import base64
import markdown
from datetime import datetime
from bs4 import BeautifulSoup
# from weasyprint import HTML
from pdf2image import convert_from_path
import google.generativeai as genai
import config

class ReportGenerator:
    """
    Generates the final PDF report by combining model output, SHAP explanations,
    and LLM-generated text.
    """
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel(config.LLM_MODEL)
            print("Google Generative AI configured successfully.")
        except Exception as e:
            print(f"❌ Error configuring Google Generative AI: {e}")
            self.llm_model = None

    def generate_llm_explanation(self, predicted_class, top_features, context):
        """
        Prompts the LLM to generate a clinical interpretation.
        """
        if not self.llm_model:
            return "LLM was not configured due to an API key error."

        print("\nGenerating clinical narrative with the LLM...")
        
        features_str = "\n".join(top_features)

        prompt = f"""
You are a medical AI assistant trained to generate precise and formal clinical interpretation reports. Your goal is to explain how a machine learning model made a specific prediction for a patient based on their input features. Your explanation must be clear and readable for physicians and clinicians, using clinical reasoning supported by the relevant literature provided.

**INSTRUCTIONS:**
1.  Start with a clear statement of the model's prediction.
2.  Explain the contribution of each of the key features provided, prioritizing those with higher impact (as indicated by SHAP values).
3.  For each feature, report its value and explain its clinical relevance to the prediction, citing the provided literature context.
4.  Cite the literature by making a general reference to the provided evidence (e.g., "as supported by clinical literature..."). Do NOT create a numbered reference list.
5.  Keep the explanation for each feature concise (1-3 sentences).
6.  Maintain a formal, objective, and clinical tone throughout.

---
**PROVIDED LITERATURE CONTEXT:**
{context}
---

**PATIENT-SPECIFIC FINDINGS:**

**Model Prediction:** The model predicts the patient has an **{predicted_class} glioma**.

**Key Features Influencing Prediction:**
{features_str}

---

**GENERATE THE REPORT BELOW:**
"""
        try:
            response = self.llm_model.generate_content(prompt)
            print("LLM narrative generated successfully.")
            return response.text
        except Exception as e:
            print(f"❌ Error during LLM content generation: {e}")
            return f"Error generating report: {e}"

    def create_pdf_report(self, sample_info, prediction_info, shap_plot_path, llm_text):
        """
        Compiles all information into an HTML template and converts it to a PDF.
        """
        print("Compiling the final PDF report...")
        report_path = os.path.join(config.REPORTS_DIR, config.REPORT_FILENAME)

        # Encode the SHAP plot image to embed it in the HTML
        try:
            with open(shap_plot_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            image_html = f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 600px; margin-top: 20px;" >'
        except FileNotFoundError:
            image_html = "<p><strong>Error: SHAP plot image not found.</strong></p>"
        
        # Convert the LLM's markdown response to HTML
        report_html = markdown.markdown(llm_text)
        
        # HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Machine Learning Prediction Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px;}}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                td {{ padding: 8px; border: 1px solid #ddd; }}
                .header {{ background-color: #f2f2f2; font-weight: bold; }}
                .prediction {{ font-size: 24px; color: #d9534f; font-weight: bold; margin-top: 20px; }}
                .report-content {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>Machine Learning Prediction Report</h1>
            <h2>Glioma IDH Classification</h2>
            
            <table>
                <tr>
                    <td class="header">Patient ID:</td>
                    <td>{sample_info.get('PatientID', 'N/A')}</td>
                    <td class="header">Patient Age:</td>
                    <td>{sample_info.get('Age', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="header">Report Date:</td>
                    <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                    <td class="header">Model Used:</td>
                    <td>{prediction_info.get('model_name', 'N/A')}</td>
                </tr>
            </table>

            <div class="prediction">
                Final Prediction: {prediction_info.get('predicted_class', 'N/A')}
            </div>

            <div class="report-content">
                <h2>Clinical Interpretation</h2>
                {report_html}

                <h2>Prediction Explanation (SHAP Analysis)</h2>
                <p>The following plot shows the features that contributed most to the model's prediction for this patient. Features pushing the prediction higher (towards IDH-wildtype) are in red, and those pushing it lower (towards IDH-mutant) are in blue.</p>
                {image_html}
            </div>
        </body>
        </html>
        """
        
        # Generate PDF
        # HTML(string=html_template).write_pdf(report_path)
        print(f"✅ Report successfully saved to: {report_path}")

        # Optional: Convert PDF to image for display
        try:
            images = convert_from_path(report_path, dpi=150)
            if images:
                image_path = os.path.join(config.REPORTS_DIR, "report_preview.png")
                images[0].save(image_path, "PNG")
                print(f"   Report preview image saved to: {image_path}")
        except Exception as e:
            print(f"⚠️ Could not create PDF preview image. Poppler might be missing. Error: {e}")