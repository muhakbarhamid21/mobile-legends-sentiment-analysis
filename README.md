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
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
      - [Sentiment Distribution](#sentiment-distribution)
      - [Word Count Analysis](#word-count-analysis)
      - [Word Cloud](#word-cloud)
      - [Unique Word Count Analysis](#unique-word-count-analysis)
      - [Sentiment Proportion by Rating](#sentiment-proportion-by-rating)
      - [Sentiment Over Time](#sentiment-over-time)
      - [Top Frequent Words](#top-frequent-words)
    - [Results](#results)
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

### Exploratory Data Analysis (EDA)

#### Sentiment Distribution

![Sentiment Distribution](https://github.com/user-attachments/assets/5633bec2-9f1d-43f5-859d-ffb8d1491ead)

#### Word Count Analysis

![Word Count Analysis 2](https://github.com/user-attachments/assets/8a75f629-f6d8-4d13-99b8-facad0b287de)
![Word Count Analysis 1](https://github.com/user-attachments/assets/feac79fa-d44c-4f77-bb3e-83b00e72b656)

#### Word Cloud

![Word Cloud Positive](https://github.com/user-attachments/assets/b9b303d1-8a4b-4b1b-8f8d-354d062c6537)
![Word Cloud Neutral](https://github.com/user-attachments/assets/2623428d-eaa7-4d91-b423-b45a2eb13083)
![Word Cloud Negative](https://github.com/user-attachments/assets/71eb6ede-f008-4f9c-80ec-3c2b4220740b)

#### Unique Word Count Analysis

![Unique Word Count Analysis](https://github.com/user-attachments/assets/a35999e3-e429-4108-aea5-7623ca09d4c1)

#### Sentiment Proportion by Rating

![Sentiment Proportion by Rating](https://github.com/user-attachments/assets/8192a1f3-aa2b-4baf-a3fe-1960d8304884)

#### Sentiment Over Time

![Sentiment Over Time](https://github.com/user-attachments/assets/0d6a84b0-2fe6-4107-9bbd-4774d89ecfca)

#### Top Frequent Words

![Top Frequent Words Positive](https://github.com/user-attachments/assets/cfc45ba8-23bf-4a1d-9ecd-43a9ac009d98)
![Top Frequent Words Neutral](https://github.com/user-attachments/assets/8add7505-b04b-4468-8a1a-b1f399fc5b19)
![Top Frequent Words Negative](https://github.com/user-attachments/assets/57519a18-1e28-43a6-b1fa-a6b167cf0de2)

### Results

![comparison_of_training_and_testing_accuracy](https://github.com/user-attachments/assets/374ea63e-203d-4bf8-90e2-17afa5f17015)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
