import streamlit as st
import pandas as pd
import os
from PIL import Image

if __name__=='__main__':
    path = os.path.dirname(__file__)
    csv_path = path+'/bank_account_fraud.csv'
    class_dist = pd.read_csv(csv_path,index_col=0)
    
    image_path = path+'/bank_acccount_fraud_confusion_matrix.png'
    image = Image.open(image_path)

    st.set_page_config(page_title="Bank Account Fraud Detection")
    st.header("Bank Account Fraud Detection")
    st.markdown("""
                This dataset was obtained from [kaggle]( https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022). 
                The dataset has been synthetically generated to mimic fraudulent transactions, anf published at  NeurIPS 2022. There are a total of 6 datasets -
                a base dataset, and 5 variations on it. I used the base dataset.  
                There are 1 million instances out of which 1.1% are fraud.""")
    
    st.dataframe(data=class_dist)
    
    st.markdown("""There are a total of 30 features - numerical and categorical.  \n I re-sampled the data, by taking samples of 
                not-fraud class the same as the number of fraud instances in the training dataset. I trained CatBoost classifier 
                and achieved a recall score of 82%. 
                """)
    st.markdown("----")
    st.markdown("""
                **Performance Metrics**  \n
                - Recall score of **0.82** (model accurately predicts 82\% of frauds )  \n
                - ROC-AUC score of **0.897**  \n
                - The False Positive Rate is **0.18**, i.e. 18\% of non-fraudulent transactions are flagged as fraud.
                """)
    st.markdown("""This model has mediocre performance - there is a lot of scope of improvement to reduce the errors in prediction for both classes.""")
    st.markdown("----")
    st.markdown("""Here is the confusion matrix for the model predictions on the whole dataset.""")
    st.image(image)