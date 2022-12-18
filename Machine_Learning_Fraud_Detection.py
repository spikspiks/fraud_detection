import streamlit as st
import os

if __name__=='__main__':
    st.set_page_config(page_title="Fraud Detection through Machine Learning")
    st.header("Fraud Detection through Machine Learning")
    st.markdown("""
                Modern financial platforms conduct thousands of transactions every minute. 
                Unscrupulous actors are always lurking in these platforms to conduct fraudulent transactions - 
                through stolen credit card information, stolen online banking credentials, or any other methods. 
                
                Machine learning is the ideal tool to detech these fraudulent transactions. This is a binary classification problem - fraud or not fraud.
                
                The problem one faces is that fraudulent transactions are rare events. Hence, these datasets are
                highly imbalanced, i.e., number of fraud instances are typically around 1\% or less of
                the total number of transactions. This is a challenge for most machine learning algorithms from the get go,
                as there are too few instances of the fraud for the algorithm to learn to predict them all. 
                
                I try to get around this by re-sampling the dataset. I under-sample the majority class to equalize the class distributions 
                in the training data. 
                
                Once a model is trained, it is important to choose the appropriate metric to evaluate the model. Accuracy is a terrible metric
                to evaluate the model on, due to the heavy class imbalance. Recall, F1-score, ROC-AUC Score and Confusion Matrix are the metrics I use to 
                evaluate the model performance. The Confusion Matrix is especially important, as it lets one see easily how many instances are being 
                incorrectly labelled. Number of false negatives, i.e., frauds classified as not fraud, is crucial here - you don't want to let fraud go 
                undetected. But you cannot have too many false positives either, i.e., regular transactions classified as fraud - that would mean
                legitimate transactions would be flagged and stopped, much to the dissatisfaction of users.
            
                This is a showcase of these techniques. I obtained 4 banking and credit card fraud datasets from kaggle.com, resampled them 
                trained tree-based gradient boosting machine learning algorithms like CatBoost, LightGBM and XGBoost.
                """)
    st.markdown("----")
    st.write("Check out my [GitHub](https://github.com/spikspiks/fraud_detection) for jupyter notebooks showing how the models were trained.")