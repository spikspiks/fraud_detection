import streamlit as st
import pandas as pd
import os
from PIL import Image

if __name__=='__main__':
    path = os.path.dirname(__file__)
    csv_path = path+'/online_fraud.csv'
    
    image_path = path+'/online_fraud_confusion_matrix.png'
    image = Image.open(image_path)
    
    class_dist = pd.read_csv(csv_path,index_col=0)
    st.set_page_config(page_title="Online Payment Fraud Detection")
    st.header("Online Payment Fraud Detection")
    st.markdown("""
                This dataset was obtained from [kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection). 
                There are 6.35 million total instances with only 8213 instances of fraud (0.13% - very highly imbalanced).""")
    
    st.dataframe(data=class_dist)
    
    st.markdown("""There aren't too many features and they include time of transaction, type of transaction, 
                starting and final balance of customer starting the transaction, and starting and final balance of recipient.  \n\nTo 
                create the training and testing datasets, I take same number of samples from Not Fraud class as the number of total fraud
                instances - 8213. I create training and test datasets from this, and train a CatBoost classifier model.""")
    st.markdown("----")
    st.markdown("""
                **Performance Metrics** \n
                - Recall score of **0.997** (model accurately predicts 99.7\% of frauds )  \n
                - ROC-AUC score of **0.999**  \n
                - F1-score of **0.999**  \n  
                - The False Positive Rate is **0.01**, i.e. 1\% of non-fraudulent transactions are flagged as fraud. This is a metric where there is room for improvement.
                """)
    st.markdown("----")
    st.markdown("""
                Since I sampled a small fraction of the not fraud instances to train the model, I got the model predictions on the whole dataset - to see
                how the model performs on the large number of not fraud instances not seen by it during both training and test phases.
                The performance metrics are almost identical.  \n
                The Confusion matrix shows that the false negative rate is **0.3%** and false positive rate is **1.2%**. 
                """)
    st.image(image)