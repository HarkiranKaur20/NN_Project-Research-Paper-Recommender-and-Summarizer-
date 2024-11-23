# NN_Project-Research-Paper-Recommender-and-Summarizer-

This project is a **Research Paper Recommendation and Summarization System** that helps users find similar research papers based on a provided title and generates concise summaries of their abstracts. It integrates machine learning techniques and provides an interactive web application for ease of use.

---

## **Features**

1. **Research Paper Recommendation**:
   - Recommends research papers similar to a user-provided title using a **TF-IDF similarity model**.
   
2. **Abstract Summarization**:
   - Generates concise summaries of research paper abstracts using a **fine-tuned T5 Transformer model**.

3. **Interactive Web Application**:
   - Developed with **Streamlit**, enabling users to interact with the system easily.

---

## **Project Structure**

```plaintext
📂 project-root/
├── arxiv-t5-model/               # Directory containing the fine-tuned T5 model
├── app.py                        # Streamlit web application
├── requirement.txt               # Python dependencies
├── summarization_model.pkl       # Pickle file for the summarization model
├── tfidf_matrix.pkl              # TF-IDF matrix for paper titles
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer object
├── titles_abstracts.pkl          # Titles and abstracts of research papers
├── recommendation.ipynb          # Jupyter notebook for the recommendation model
├── summarization.ipynb           # Jupyter notebook for the summarization model
└── filtered_arxiv_data.json      # Filtered dataset of research papers
```

---

## **Installation and Setup**

### **Prerequisites**

- **Python 3.8 or later** is required.
- Ensure `pip` is installed.

### **Steps to Run**

1. Clone the repository or download the project files:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirement.txt
   ```

3. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

4. Open the URL displayed in the terminal (e.g., `http://localhost:8501`) to interact with the app.

---

## **File Descriptions**

### **Main Files**
- **`app.py`**:
  - Contains the code for the Streamlit web app, integrating recommendation and summarization functionalities.

- **`requirement.txt`**:
  - Includes all Python dependencies required to run the project.

- **`arxiv-t5-model/`**:
  - Directory with the fine-tuned T5 Transformer model used for abstract summarization.

- **`summarization_model.pkl`**:
  - Pickle file of the pre-trained summarization model.

- **`tfidf_matrix.pkl`**:
  - Precomputed TF-IDF matrix for research paper titles.

- **`tfidf_vectorizer.pkl`**:
  - Vectorizer object trained using the research paper dataset.

- **`titles_abstracts.pkl`**:
  - Pickle file containing preprocessed titles and abstracts.

- **`filtered_arxiv_data.json`**:
  - Dataset of research papers filtered by category (e.g., AI, ML, CV).

### **Jupyter Notebooks**
- **`recommendation.ipynb`**:
  - Notebook for building and testing the recommendation model using TF-IDF.
  - Processes the dataset and calculates similarity scores.

- **`summarization.ipynb`**:
  - Notebook for fine-tuning the T5 model and summarizing research paper abstracts.

---

## **Functionality**

### **1. Paper Recommendation**
- Computes the similarity between research paper titles using **TF-IDF** and cosine similarity.
- Recommends the top N papers based on the similarity score.

### **2. Abstract Summarization**
- Generates concise summaries of paper abstracts using the fine-tuned T5 model.
- Summaries are optimized for clarity and brevity.

### **3. Streamlit Web App**
- Input a research paper title to:
  - Get a list of similar papers.
  - Generate summaries of recommended papers.

---

## **How to Use**

### **1. Running the Web Application**
- Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```
- Enter a research paper title in the input box to:
  - View recommended papers based on similarity.
  - Summarize their abstracts.

### **2. Using the Notebooks**
- Open **`recommendation.ipynb`** to:
  - Train or test the recommendation model.
  - Recompute the TF-IDF matrix and vectorizer if the dataset is updated.

- Open **`summarization.ipynb`** to:
  - Fine-tune the T5 model or update summaries.

---

## **Dataset**

The dataset is sourced from [arXiv Metadata](https://www.kaggle.com/Cornell-University/arxiv) and includes papers in categories like:
- Artificial Intelligence (`cs.AI`)
- Machine Learning (`cs.LG`)
- Computer Vision (`cs.CV`)

Each paper in the dataset has:
- **Title**
- **Abstract**
- **Categories**

The dataset is preprocessed and stored in `titles_abstracts.pkl` and `filtered_arxiv_data.json`.

---

## **Dependencies**

Key dependencies are listed in `requirement.txt` and include:
- `streamlit`: For building the web application.
- `transformers`: For handling the T5 Transformer model.
- `scikit-learn`: For TF-IDF vectorization and similarity calculations.
- `sentencepiece`: Tokenizer library required for T5.
- `pickle`: For saving/loading trained models and datasets.

To install all dependencies:
```bash
pip install -r requirement.txt
```

---

## **Model Training**

### **TF-IDF Training**
- **Notebook**: `recommendation.ipynb`
- **Steps**:
  1. Preprocess the research paper dataset.
  2. Compute the TF-IDF matrix for paper titles.
  3. Save the vectorizer and matrix as `.pkl` files.

### **T5 Model Fine-Tuning**
- **Notebook**: `summarization.ipynb`
- **Steps**:
  1. Fine-tune the T5 model on research paper abstracts.
  2. Save the fine-tuned model in the `arxiv-t5-model` directory.

---

## **Contributing**

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch-name
   ```
5. Open a pull request for review.

---

## **Acknowledgements**

- Dataset from [arXiv Metadata](https://www.kaggle.com/Cornell-University/arxiv).
- T5 model from [Hugging Face](https://huggingface.co/).
- Streamlit for building the web application.
