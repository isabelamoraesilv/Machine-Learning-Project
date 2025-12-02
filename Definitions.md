# Definitions and Explanations for Project Concepts

Here are the definitions and explanations for the concepts and algorithms used in your project.

## 1. The Task

* **Supervised Binary Classification Task:** This is a type of machine learning problem where the goal is to categorize data into one of two mutually exclusive classes (e.g., "Suicide" vs. "Non-Suicide"). It is called "supervised" because the algorithm learns from a training dataset where the correct answers (labels) are already provided.

## 2. Preprocessing & Dimensionality

* **Regex (Regular Expressions):** A specialized sequence of characters that forms a search pattern. In Data Science, regex is a powerful tool used during text preprocessing to identify and manipulate specific string formats, such as finding and removing URLs, HTML tags, or punctuation from raw text.
* **Truncated SVD (Singular Value Decomposition):** A dimensionality reduction technique commonly used for sparse matrices (like those created by TF-IDF). It works by factoring the data matrix to identify the most significant underlying patterns (latent concepts) and discarding the "noise." Unlike PCA, it does not require centering the data, making it efficient for large text datasets.
* **Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique used to simplify complex datasets while retaining the most significant patterns and trends. It works by mathematically transforming a set of correlated variables into a smaller set of uncorrelated variables called **Principal Components**. Imagine a cloud of data points in a 3D space; PCA finds the "best angle" to squash this data onto a 2D sheet (or even a 1D line) so that the data is spread out as much as possible. By maximizing the **variance** (the spread) along these new axes, PCA reduces noise and computational cost, making the data easier for algorithms to process and for humans to visualize, with minimal loss of information.



## 3. Algorithms & Models

* **Logistic Regression:** Despite its name, this is a linear classifier, not a regression algorithm. It calculates the weighted sum of input features and passes the result through a "sigmoid" function. This maps the output to a probability between 0 and 1, allowing the model to decide which class an observation belongs to based on a probability threshold (usually 0.5).
* **Multinomial Naïve Bayes:** A probabilistic classifier based on Bayes' theorem. It assumes that features (words) are independent of one another. The "Multinomial" variant is specifically designed for data representing discrete counts, making it a standard baseline for text classification tasks involving word frequencies.
* **Linear SVM (Support Vector Machine):** A classifier that aims to find the best linear boundary (a hyperplane) that separates data points of two classes. It tries to maximize the "margin" (the distance) between the boundary and the nearest data points of each class.
* **MLP (Multi-Layer Perceptron):** A type of feedforward Artificial Neural Network. It consists of an input layer, one or more "hidden" layers of neurons with non-linear activation functions, and an output layer. Through training (backpropagation), it learns to model complex, non-linear relationships in the data that simpler linear models cannot capture.

## 4. Ensemble Methods

* **Random Forest:** An ensemble learning method that constructs a multitude of decision trees during training. Each tree looks at a random subset of the data and a random subset of features. The final output is determined by "voting"—averaging the predictions of all the individual trees to improve accuracy and control overfitting.
* **Gradient Boosting:** An ensemble technique that builds models sequentially rather than in parallel. It starts with a weak model and adds new models one by one, where each new model specifically tries to correct the errors (residuals) made by the previous combined models.
* **Bagging (Bootstrap Aggregating):** A technique to improve stability and reduce variance. It creates multiple versions of a training set by random sampling with replacement. A separate model is trained on each sample, and their predictions are averaged.
* **Meta-estimator:** A high-level algorithm that takes another algorithm (the "base" estimator) as an input and enhances its behavior. For example, in your project, the `BaggingClassifier` is a meta-estimator that wraps around the `LogisticRegression` (base estimator) to train it multiple times on different data subsets.

## 5. Validation & Tuning

* **Cross-Validation:** A resampling procedure used to evaluate a model's performance on limited data. It ensures that every observation is used for both training and testing. This prevents the model from getting a "lucky" score just because of how the data was split.
* **Fold:** In Cross-Validation, the data is divided into $k$ equal groups called "folds." If you do 5-fold CV, the process runs 5 times: each time, a different fold is used as the test set, while the remaining 4 folds form the training set.
* **GridSearchCV:** An automated method for hyperparameter tuning. You define a "grid" of possible values for model parameters (e.g., regularization strength). The algorithm creates models for every possible combination in the grid and uses Cross-Validation to determine which combination gives the best performance.

## 6. Metrics

* **Accuracy:** The most intuitive performance metric. It measures the ratio of correctly predicted observations to the total observations. It answers: "What percentage of the time is the model correct?"
* **Recall (Sensitivity):** A metric that measures the ability of a model to find all the positive cases. In your context, it answers: "Out of all the actual suicide-related posts, what percentage did the model correctly identify?" (High recall minimizes False Negatives).
* **F1-Score:** The harmonic mean of Precision and Recall. It provides a single score that balances the trade-off between the two. It is more reliable than accuracy when you need to balance avoiding false alarms (Precision) with finding all cases (Recall).
