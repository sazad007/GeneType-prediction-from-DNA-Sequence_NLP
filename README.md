# Gene Type Prediction from DNA Sequences using NLP

This project explores how **Natural Language Processing (NLP)** techniques can be applied to biological data for predicting **gene types** from DNA sequences. Using the [DNA Sequence Prediction Dataset](https://www.kaggle.com/datasets/harshvardhan21/dna-sequence-prediction/data), we build a machine learning pipeline that combines sequence embeddings, gene metadata, and classification models.

---

## 📌 Dataset
We use the Kaggle dataset:

- **Train / Validation / Test CSVs** (`train.csv`, `validation.csv`, `test.csv`)  
- Key columns:
  - `NucleotideSequence`: DNA sequence (e.g., `"AGCTTAGC..."`)
  - `SequenceLength`: Length of the DNA sequence
  - `Symbol` & `Description`: Gene metadata
  - `GeneGroupMethod`: Grouping method
  - `GeneType`: **Target variable**

---

## ⚙️ Approach
The workflow involves both **NLP-inspired preprocessing** and **machine learning classification**:

1. **Data Cleaning**
   - Remove unwanted characters with regex
   - Handle missing values (`dropna`)

2. **Feature Engineering**
   - **DNA sequence** → TF-IDF Vectorizer with character-level trigrams (`ngram_range=(3,3)`)
   - **Description & Symbol** → TF-IDF (bag of words)
   - **Sequence length** → StandardScaler
   - **GeneGroupMethod** → Label Encoding

3. **Feature Fusion**
   - Concatenate sequence vectors, metadata vectors, and numeric features using `numpy` & `scipy.sparse.hstack`

4. **Modeling**
   - Neural Network with `Dense` layers
   - Softmax output for multiclass classification
   - Loss: `sparse_categorical_crossentropy`
   - Optimizer: `adam`

5. **Evaluation**
   - Accuracy score on validation and test data
   - Top predicted class + probability

---

## 📊 Results

- **Training Accuracy:** ~99.28%  
- **Validation Accuracy:** ~99.30%  
- **Test Accuracy:** **99.05%**

Sample predictions from the test set show highly confident classification scores for multiple categories (features from test.csv):

| Index | Label             | Confidence |
|-------|-------------------|------------|
| 0     | BIOLOGICAL_REGION | 0.9999999  |
| 1     | snRNA             | 0.9907     |
| 4     | PSEUDO            | 0.9999919  |
| 9     | PROTEIN_CODING    | 0.9996306  |
| 16    | ncRNA             | 0.9997390  |
| 43    | snoRNA            | 0.9179     |
| 47    | snoRNA            | 0.9837     |


The model demonstrates excellent generalization, achieving **>99% accuracy** across training, validation, and test splits.

---

## 🧬 Output

- High classification accuracy across all gene types.  
- Model predictions are confident, with most probabilities exceeding **0.99**.  

---

## 🔬 Key Learnings
- DNA sequences can be treated as **text data**, where k-mers (e.g., 3-character n-grams) capture biological motifs.
- Combining **sequence features + metadata features** improves classification performance.
- NLP techniques such as **TF-IDF** are effective in biological sequence modeling.


