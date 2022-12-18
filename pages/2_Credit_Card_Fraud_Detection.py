import streamlit as st
import pandas as pd
import os
from PIL import Image

if __name__=='__main__':
    path = os.path.dirname(__file__)
    csv_path = path+'/credit_card_fraud.csv'
    class_dist = pd.read_csv(csv_path,index_col=0)
    
    image_path = path+'/cc_fraud_confusion_matrix.png'
    image = Image.open(image_path)

    st.set_page_config(page_title="Credit Card Fraud Detection")
    st.header("Credit Card Fraud Detection")
    st.markdown("""
                This dataset was obtained from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
                The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
                There are 492 frauds out of 284,807 transactions that occured over 2 days (0.13% - very highly imbalanced).""")
    
    st.dataframe(data=class_dist)
    
    st.markdown("""The data has been transformed into purely numerical quantities to preserve confidentiality of the original data.\n\n Initially I used the whole dataset to create training and test datasets and trained some
                tree based machine learning models: Decision Tree, Random Forest, CatBoost, XGBoost and LightGBM. The best model out of those, XGBoost,
                achieved a recall score of 80%.  \n I then re-sampled the data, by taking samples of not-fraud class the same as the number of fraud
                instances - 492. This improved the recall score to 92% (CatBoost classifier). 
                """)
    st.markdown("----")
    st.markdown("""
                **Performance Metrics**:  \n
                - Recall score of **0.92** (model accurately predicts 92\% of frauds )  \n
                - ROC-AUC score of **0.985**  \n
                - The False Positive Rate is **0.027**, i.e. 2.7\% of non-fraudulent transactions are flagged as fraud. This is a metric where there is room for improvement.
                """)
    st.markdown("----")
    st.markdown("""
                Since I sampled a small fraction of the not fraud instances to train the model, I got the model predictions on the whole dataset to see
                how the model performs on the large number of not fraud instances not seen by it during both training and test phases. Performance is pretty
                spectacular.
                The Confusion matrix shows that the false negative rate is **1.4%** and false positive rate is **1.8%**.  
                """)
    st.image(image)