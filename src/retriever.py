import os
import json
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config

class PaperRetriever:
    """
    Handles fetching paper metadata from Europe PMC.
    (Note: Downloading and parsing PDFs is a heavy task and is simplified here)
    """
    def fetch_paper_metadata(self):
        print("\nFetching paper metadata from Europe PMC...")
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": config.PUBMED_QUERY,
            "format": "json",
            "pageSize": config.MAX_PUBMED_RESULTS
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            
            papers = {}
            for paper in results:
                title = paper.get("title", "").strip()
                # A simple way to create a filename-safe title
                filename = "".join([c for c in title if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
                if title:
                    papers[filename] = paper.get("doi", "")
            
            print(f"Found metadata for {len(papers)} papers.")
            return papers
        except requests.RequestException as e:
            print(f"❌ Error fetching from Europe PMC: {e}")
            return {}

class FaissRetriever:
    """
    Builds a FAISS vector store from text and retrieves relevant context.
    """
    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME):
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.doc_ids = []

    def build_index_from_texts(self, paper_texts):
        """
        Embeds documents and builds a FAISS index.
        """
        print("\nBuilding FAISS vector store from paper texts...")
        
        for name, content in paper_texts.items():
            self.documents.append(content)
            self.doc_ids.append(name)
            
        print(f"Embedding {len(self.documents)} documents...")
        embeddings = self.model.encode(self.documents, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_context(self, features_list, k=config.RETRIEVER_TOP_K):
        """
        Retrieves relevant text snippets for a list of feature names.
        """
        if self.index is None:
            return "Retriever index has not been built."
            
        print("\nQuerying FAISS index for relevant literature context...")
        all_retrieved_text = ""
        
        # The features from the explainer are formatted with markdown, clean them first
        clean_features = [f.split(':')[0].replace('*', '').strip() for f in features_list]
        
        query_text = ", ".join(clean_features)
        
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        retrieved_docs = [self.documents[i] for i in indices[0]]
        retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        
        print(f"Retrieved top {k} documents based on features: {retrieved_ids}")
        
        # For simplicity, we concatenate the full text of the top k documents.
        # A more advanced approach would be to chunk and retrieve specific sentences.
        all_retrieved_text = "\n\n---\n\n".join(retrieved_docs)
        
        return all_retrieved_text

# --- Helper function to simulate text extraction ---
# In a real scenario, this would involve downloading and parsing hundreds of PDFs,
# which is slow and error-prone. We will simulate by creating a dummy text file.
def get_paper_texts(force_create=False):
    """
    Simulates the process of extracting text from all papers.
    If a cached file exists, it loads it. Otherwise, it creates a dummy file.
    """
    if os.path.exists(config.ALL_TEXTS_PATH) and not force_create:
        print(f"Loading cached paper texts from {config.ALL_TEXTS_PATH}...")
        with open(config.ALL_TEXTS_PATH, 'r') as f:
            return json.load(f)
    else:
        print("Simulating paper download and text extraction...")
        # This is a placeholder. In a real workflow, you would download PDFs
        # from the DOIs and extract their text.
        dummy_texts = {
            "Paper_on_Age_and_Glioma": "Age is a significant prognostic factor in glioma. Older patients, typically over 55, are more likely to have IDH-wildtype glioblastoma, which is associated with a poorer prognosis. In contrast, IDH-mutant gliomas are more common in younger patients.",
            "Paper_on_MGMT_in_Glioblastoma": "O6-methylguanine-DNA methyltransferase (MGMT) promoter methylation is a key biomarker in glioblastoma. Methylation silences the gene, leading to better response to temozolomide chemotherapy. MGMT methylation is more frequently observed in IDH-mutant gliomas.",
            "Paper_on_Radiomics_Sphericity": "Radiomic features quantify tumor characteristics from medical images. Tumor sphericity, a measure of how spherical a tumor is, has been linked to prognosis. Lower sphericity, indicating a more irregular shape, is often associated with more aggressive, infiltrative tumors like IDH-wildtype glioblastoma.",
            "Paper_on_GLCM_Features": "Gray-Level Co-occurrence Matrix (GLCM) features describe tumor texture. High cluster prominence can indicate heterogeneity and structural complexity within the tumor, which may be characteristic of high-grade gliomas."
        }
        with open(config.ALL_TEXTS_PATH, 'w') as f:
            json.dump(dummy_texts, f)
        print(f"Saved dummy paper texts to {config.ALL_TEXTS_PATH}")
        return dummy_texts