# Email-Spam-Classifier
A machine learning pipeline that classifies SMS/email messages as Spam or Ham (not spam) using text vectorization and a Multinomial Naive Bayes classifier — with full EDA, evaluation, and a ready-to-use prediction function.
This project builds an end-to-end spam detection system on the classic spam.csv dataset. It covers data cleaning, exploratory analysis (class distribution and word clouds), model training with a scikit-learn pipeline, and a thorough evaluation suite (ROC-AUC, confusion matrices, classification reports) — before wrapping everything into a simple detect_spam() function for real-time predictions.

✨ Features


Data cleaning — renames raw columns (v1, v2 → Category, Message), drops unnamed junk columns, and encodes labels (spam → 1, ham → 0)
Exploratory Data Analysis

Pie chart showing the spam vs. ham class distribution
Word cloud highlighting the most frequent words in spam messages



Modeling — a scikit-learn Pipeline combining CountVectorizer (bag-of-words) with MultinomialNB
Evaluation — a reusable evaluate_model() function that reports:

ROC-AUC score (train & test) with ROC curve plot
Confusion matrices (train & test)
Full classification reports (precision, recall, F1)



Inference — a detect_spam(email_text) function that returns a human-readable prediction for any input message
