# Software Defect Prediction Using Feature Selection and Machine Learning

## Abstract

Software testing is crucial for delivering high-quality software. Early identification of defects minimizes costs and reduces risks associated with late-stage detection. Software Defect Prediction (SDP) techniques leverage machine learning approaches to predict defect-prone modules early in the development cycle. Feature selection (FS) plays a significant role in SDP model performance. This project evaluates the impact of various FS techniques on four supervised learning classifiers (SVM, KNN, Decision Tree, Naive Bayes) applied to six NASA datasets from the PROMISE repository.

## Introduction

Defects in software can lead to severe consequences, so early detection is paramount. SDP identifies modules with a higher likelihood of containing defects, enabling developers to prioritize testing efforts. This project investigates the effectiveness of FS techniques in conjunction with machine learning algorithms for enhancing SDP model performance.

## Methodology

**1. Data Preprocessing:**

* Import necessary libraries (NumPy, pandas, matplotlib, seaborn).
* Load CSV datasets using pandas.
* Perform data exploration (shape, head, information).
* Apply MinMaxScaler for normalization.
* Separate features (X) and target variable (defects) for modeling.

**2. Feature Selection:**

* Evaluate various FS techniques:
    * **Filter-based:**
        * SelectKBest (mutual information)
        * SelectPercentile (mutual information)
    * **Wrapper-based (not implemented in this code, see future work):** Utilize feature importance scores from trained models to select features.
    * **Embedded:** Feature selection is integrated within the model training process (e.g., recursive feature elimination). 
* Select a subset of features based on FS technique output.

**3. Model Training and Evaluation:**

* Split data into training and testing sets using `train_test_split`.
* Implement four machine learning classifiers:
    * SVM (Support Vector Machine)
    * KNN (K-Nearest Neighbors)
    * Decision Tree
    * Naive Bayes
* Train each model on the training set.
* Evaluate model performance on the testing set using metrics like accuracy, confusion matrix, and classification report.

## Results

- The project demonstrate that FS techniques significantly improve model performance.
- SVM achieved the best overall accuracy among the evaluated classifiers.
- Fisher's score (filter-based method) appeared to be the most effective FS technique in this study.

## Conclusion

This project highlights the importance of FS in software defect prediction. By optimizing feature selection, we can build more reliable and accurate SDP models, leading to improved software quality.

## Code

The provided code demonstrates the basic structure for feature selection, training, and evaluation of machine learning models for software defect prediction. Further refinements can be made to incorporate additional FS techniques, hyperparameter tuning, and more comprehensive analysis.
