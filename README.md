### **Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data**

**Student Information**

  * **Name:** Kashyap Shankar Iyer
  * **Roll Number:** DA25C012

-----

**Documentation**

This submission consists of a single Jupyter Notebook file containing all code, visualisations, and analysis for the assignment. Its purpose is to explore advanced oversampling techniques to address severe class imbalance in the Credit Card Fraud Detection dataset. This notebook builds upon previous methods by implementing and evaluating Gaussian Mixture Model (GMM)-based oversampling, comparing its effectiveness against a baseline Logistic Regression classifier.

-----

**Folder Structure**

```
.
└── Kashyap_DA25C012_DA5401_Assignment_4_GMM_Resampling.ipynb
```

-----

**Dataset Information**

The analysis in this notebook is based on the **Credit Card Fraud Detection** dataset, sourced from Kaggle. The dataset contains transactions made by European cardholders and features 28 principal components (`V1` to `V28`), along with `Time` and `Amount`. The target class is highly imbalanced, indicating whether a transaction is fraudulent (1) or legitimate (0). [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

-----

**Notebook Structure and Content**

The notebook is organised into the following sections to guide the reader through the analysis:

  * **Part A: Data Exploration and Baseline Model**
    This section covers loading the dataset, performing an initial exploratory analysis to confirm the severe class imbalance, and establishing a baseline performance. The baseline is a standard Logistic Regression model trained on the original, imbalanced data.

  * **Part B: Resampling Approaches**
    This is the core section where different techniques are applied to balance the training data. It involves:

    1.  **GMM-Based Oversampling:** A more sophisticated approach where a Gaussian Mixture Model is first fitted to the minority (fraud) class. New synthetic samples are then generated from the learned GMM distribution, aiming to create higher-quality, more representative data than SMOTE.
    2.  **Hybrid Technique (CBU + GMM):** Implementing a hybrid approach that combines Clustering-Based Undersampling (CBU) on the majority class with GMM-based oversampling on the minority class to achieve a balanced dataset.

  * **Part C: Model Comparison and Analysis**
    This section evaluates the impact of each resampling technique on the classification task. Four models are built and compared:

    1.  A **Baseline Model** using Logistic Regression on the original, imbalanced data.
    2.  A **SMOTE-Enhanced Model** trained on the data from SMOTE.
    3.  A **GMM Oversampling-Enhanced Model** trained on the GMM-generated data.
    4.  A **Hybrid (CBU + GMM)-Enhanced Model** trained on the hybrid-sampled data.

    **Comparison and Analysis:** The notebook concludes by summarizing the performance metrics (Precision, Recall, F1-Score) of the four models in a comparative table and with visualisations. It provides a detailed analysis of the trade-offs, focusing on the pros and cons of going with GMM-based techniques for such use cases.