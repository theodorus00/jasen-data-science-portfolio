"""House-price panels for the existing Day 50 portfolio app."""
import csv
import io
import json
import pandas as pd
import streamlit as st
from .src.config import PROJECT_ROOT, TRAIN_PATH, TEST_PATH
from .src.predict_notebook import FEATURES, load_notebook_artifact, predict_houses

REPORTS = PROJECT_ROOT / 'reports/notebook_method'


def artifact_signature():
    paths = [TRAIN_PATH, TEST_PATH, PROJECT_ROOT/'models/notebook_method/elasticnet_bundle.joblib']
    return tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)


@st.cache_resource
def cached_house_artifact(signature):
    return load_notebook_artifact()


def read_uploaded_csv(raw):
    """Check headers before pandas auto-renames duplicate columns."""
    text = raw.decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names are not allowed.')
    return pd.read_csv(io.StringIO(text))


def show_house_ml():
    st.header('🏠 House Price Prediction')
    st.write('Estimate property sale prices from housing characteristics using the original notebook method.')
    st.markdown('**Final model:** ElasticNet · **Compared models:** Linear Regression, Ridge, and Lasso')
    comparison = REPORTS/'model_comparison.csv'
    if comparison.exists():
        scores = pd.read_csv(comparison)
        selected = scores.loc[scores['model']=='ElasticNet'].iloc[0]
        a,b,c = st.columns(3)
        a.metric('ElasticNet CV RMSE (log)',f"{selected['cv_rmse_log_mean']:.6f}")
        b.metric('Alpha',f"{selected['alpha']:.6f}")
        c.metric('L1 ratio',f"{selected['l1_ratio']:.1f}")
        st.subheader('Notebook model comparison')
        st.dataframe(scores[['model','alpha','l1_ratio','cv_rmse_log_mean','cv_rmse_log_std']],hide_index=True)
        st.bar_chart(scores.set_index('model')[['cv_rmse_log_mean']])
        st.caption('Lower CV RMSE is better. ElasticNet remains the notebook’s final choice even if another model scores lower. These scores use preprocessing before CV and reuse folds for tuning; they are not independent holdout results.')
    else:
        st.info('Train the notebook models to display the comparison report.')
    coefficients = REPORTS/'coefficients.csv'
    if coefficients.exists():
        with st.expander('Inspect model coefficients'):
            coefs = pd.read_csv(coefficients)
            st.dataframe(coefs.head(15),hide_index=True)
            st.caption('Rows are ordered by absolute Ridge coefficient, matching the notebook. Coefficients describe the fitted model; they do not establish causality.')
    st.subheader('Predict from CSV')
    st.write('Use all 79 original features, with the original text categories. Id is optional; omit SalePrice. Empty cells follow notebook imputation rules. Up to 2,000 rows per upload.')
    with st.expander('Required columns and example'):
        st.dataframe(pd.DataFrame({'Feature':FEATURES}),hide_index=True)
        if TEST_PATH.exists():
            st.download_button('Download example CSV',TEST_PATH.read_bytes(),'house_test_example.csv','text/csv',key='house_example')
        st.download_button('Download empty template',pd.DataFrame(columns=['Id']+FEATURES).to_csv(index=False),'house_template.csv','text/csv',key='house_template')
    uploaded = st.file_uploader('Upload property CSV',type=['csv'],key='house_csv')
    if uploaded is not None:
        try:
            if uploaded.size > 10*1024*1024:
                raise ValueError('Please upload a CSV smaller than 10 MB.')
            frame = read_uploaded_csv(uploaded.getvalue())
            st.dataframe(frame.head(10),hide_index=True)
            # Cache the fitted adapter; transform only, never refit the model on upload.
            with st.spinner('Predicting house prices...'):
                state = cached_house_artifact(artifact_signature())
                predictions = predict_houses(frame,state)
            a,b = st.columns(2)
            a.metric('Properties',f'{len(predictions):,}')
            b.metric('Median predicted price',f"{predictions.SalePrice.median():,.2f}")
            st.dataframe(predictions,hide_index=True)
            st.download_button('Download house predictions',predictions.to_csv(index=False).encode('utf-8-sig'),
                               'house_price_predictions.csv','text/csv',key='house_predictions')
        except (OSError,ValueError,TypeError,KeyError,ImportError,EOFError) as error:
            st.error(f'Unable to process this file: {error}')
    st.caption('Predictions are model estimates in the dataset’s price units, not Indonesian rupiah or guaranteed market values. New categories outside the reference vocabulary are rejected.')


def show_house_eda():
    st.header('EDA — House Prices')
    try:
        df = pd.read_csv(TRAIN_PATH)
    except (OSError,ValueError) as error:
        st.error(f'Unable to load house data: {error}')
        return
    st.write('Explore labeled training properties before preprocessing. Test rows are excluded from these charts.')
    a,b,c = st.columns(3)
    a.metric('Training properties',f'{len(df):,}')
    b.metric('Raw features',len(FEATURES))
    c.metric('Median sale price',f'{df.SalePrice.median():,.0f}')
    with st.expander('Data preview and missing values'):
        st.dataframe(df.head(20),hide_index=True)
        missing = df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing[missing>0].rename('Missing values'))
    st.subheader('Sale price distribution')
    counts = pd.cut(df.SalePrice,bins=20).value_counts(sort=False)
    counts.index = counts.index.astype(str)
    st.bar_chart(counts.rename('Properties'))
    st.subheader('Property characteristics and price')
    feature = st.selectbox('Choose a property characteristic',
        ['GrLivArea','OverallQual','YearBuilt','TotalBsmtSF','GarageArea'],key='house_eda_feature')
    st.scatter_chart(df,x=feature,y='SalePrice')
    st.subheader('Median sale price by neighborhood')
    summary = df.groupby('Neighborhood').SalePrice.agg(property_count='count',median_price='median').sort_values('median_price')
    st.bar_chart(summary[['median_price']])
    st.dataframe(summary)
    st.caption('Compare neighborhood sample sizes when interpreting medians. Associations in these charts do not prove causation.')
