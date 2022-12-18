import streamlit as st
import pandas as pd
import os
from PIL import Image

if __name__=='__main__':
    path = os.path.dirname(__file__)
    csv_train_path = path+'/fraud_transactions_train.csv'
    csv_test_path = path+'/fraud_transactions_test.csv'
    class_dist_train = pd.read_csv(csv_train_path,index_col=0)
    class_dist_test = pd.read_csv(csv_test_path,index_col=0)
    
    image_path = path+'/fraud_transactions_confusion_matrix.png'
    image = Image.open(image_path)

    st.set_page_config(page_title="Fraud Transaction Detection")
    st.header("Fraud Transaction Detection")
    st.markdown("""
                This dataset was obtained from kaggle, but unfortunately, it has been taken down for unknown reasons. 
                This is a large dataset with separate test and train datasets, and is highly imbalanced.
                There are almost 1.3 million instances in the train set, and a little more than half million instances in the test set.""")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Train Set**")
        st.dataframe(data=class_dist_train)

    with col2:
        st.write("**Test Set**")
        st.dataframe(data=class_dist_test)
    
    st.markdown("""The data has a lot of features (21), but after the exploratory data analysis, I include only 5 features to train the models on.
                I have separate train and test sets from the get go, and I use all the data to train CatBoost, XGBoost and LightGBM. This gave very
                poor results - 24% recall from the best model (LightGBM).  \nI then re-sampled the data, by taking samples of not-fraud class 
                the same as the number of fraud instances in the training dataset. This improved the recall score to 85%. 
                """)
    st.markdown("----")
    st.markdown("""
                **Performance Metrics**  \n
                - Recall score of **0.85** (model accurately predicts 85\% of frauds )  \n
                - ROC-AUC score of **0.926**  \n
                - The False Positive Rate is **0.059**, i.e. 5.9\% of non-fraudulent transactions are flagged as fraud. This is a metric where there is a lot of room for improvement.
                """)
    st.markdown("----")
    st.markdown("""Here is the confusion matrix for the model predictions on both the datasets.""")
    st.image(image)