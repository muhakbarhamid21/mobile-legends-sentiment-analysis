# Mobile Legends Sentiment Analysis

This repository contains a comprehensive project for sentiment analysis on Mobile Legends reviews. The project employs various Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL) techniques to analyze the sentiment of user reviews in Indonesian. It includes feature extraction methods (TF-IDF and FastText), and multiple modeling approaches (SVM, Random Forest CNN, LSTM, and GRU). Hyperparameter tuning is performed using Optuna, and detailed evaluation and visualization of the results are provided.

## Table of Contents

- [Mobile Legends Sentiment Analysis](#mobile-legends-sentiment-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results and Visualizations](#results-and-visualizations)
  - [License](#license)

## Overview

This project is designed to explore various methods for performing sentiment analysis on Mobile Legends reviews. Key components include:

- **Data Preparation & Feature Extraction:**
  - Preprocessing and cleaning of review texts.
  - Feature extraction using TF-IDF and FastText.
  - Utilization of a colloquial Indonesian lexicon and stopword list for enhanced text processing.
- **Modeling Approaches:**
  - **Traditional Machine Learning Models:** SVM and Random Forest using scikit-learn.
  - **Deep Learning Models:** CNN, LSTM, and GRU built with TensorFlow/Keras.
- **Hyperparameter Optimization:**
  - Fine-tuning model parameters using Optuna.
- **Visualization & Evaluation:**
  - Visualizing training and testing accuracy comparisons using Plotly Express and Matplotlib.
  - Saving and reviewing model performance metrics and training histories.

## Project Structure

```bash
.
├── README.md
├── datasets
│   ├── mlbb_reviews.csv
│   └── mlbb_reviews2.csv
├── lexicon
│   └── colloquial-indonesian-lexicon.csv
├── models
│   ├── cnn
│   │   ├── best_cnn_model.h5
│   │   └── cnn_optuna_study.pkl
│   ├── fasttext_model
│   │   ├── fasttext.kv
│   │   └── fasttext.kv.vectors.npy
│   ├── gru
│   │   ├── best_gru_model.h5
│   │   └── gru_optuna_study.pkl
│   ├── indobert_model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── lstm
│   │   ├── best_lstm_model.h5
│   │   └── lstm_optuna_study.pkl
│   ├── rf
│   │   ├── best_fasttext_rf_model.pkl
│   │   ├── best_tfidf_rf_model.pkl
│   │   ├── fasttext_rf_optuna_study.pkl
│   │   └── tfidf_rf_optuna_study.pkl
│   └── svm
│       ├── best_fasttext_svm_model.pkl
│       ├── best_tfidf_svm_model.pkl
│       ├── fasttext_svm_optuna_study.pkl
│       └── tfidf_svm_optuna_study.pkl
├── notebooks
│   └── sentiment_analysis.ipynb
├── requirements.txt
├── results
│   ├── CNN_training_history.png
│   ├── GRU_training_history.png
│   ├── LSTM_training_history.png
│   └── comparison_of_training_and_testing_accuracy.png
├── scraper
│   └── playstore_scraper.ipynb
└── stoplist
    └── stopwordbahasa.csv
```

- **datasets/**: Contains CSV files with Mobile Legends reviews.
- **lexicon/**: Contains a colloquial Indonesian lexicon used for text processing.
- **models/**: Saved models and Optuna study objects for CNN, LSTM, GRU, SVM, Random Forest, and the IndoBERT model.
- **notebooks/**: Jupyter notebooks for experimentation and analysis.
- **results/**: Visualization outputs (training history plots and accuracy comparisons).
- **scraper/**: Notebooks for scraping additional review data (e.g., from Play Store).
- **stoplist/**: Contains Indonesian stopword lists.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd mobile-legends-sentiment-analysis
   ```

2. **Create a Virtual Environment (Optional):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Preparation:
   - Place your datasets (e.g., mlbb_reviews.csv and mlbb_reviews2.csv) in the datasets/ folder.
   - Ensure your dataset contains the required columns (e.g., content_clean and sentiment).
2. Feature Extraction & Modeling:
   - The project extracts features using TF-IDF and FastText.
   - It then trains various models:
     - Traditional ML Models: SVM and Random Forest are trained using both TF-IDF and FastText features.
     - Deep Learning Models: CNN, LSTM, and GRU are trained with hyperparameter tuning using Optuna.
   - To run experiments, open and execute the Jupyter notebook `notebooks/sentiment_analysis.ipynb`.
3. Scraping Additional Data:
   - Use the notebook `scraper/playstore_scraper.ipynb` to scrape additional reviews if needed.
4. Results & Visualizations:
   - Trained models and Optuna studies are saved under the respective folders in `models/`.
   - Visualizations (e.g., training history, accuracy comparisons) are saved in the `results/` folder.
   - The project also includes code to generate a grouped bar chart comparing training and testing accuracies using Plotly Express.

## Results and Visualizations

![comparison_of_training_and_testing_accuracy](https://github.com/user-attachments/assets/374ea63e-203d-4bf8-90e2-17afa5f17015)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
