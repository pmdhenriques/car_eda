import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import datetime
from glob import glob
import os

filepath = '../data/standvirtual_dataset_merge_200628.csv'
modelpath = '../models/RandomForest/rf_cv_model_200629.joblib'

@st.cache
def load_dataset(filepath):
    df = pd.read_csv(filepath)

    return df

@st.cache
def load_model(modelpath):
    clf = jl.load(modelpath)

    return clf

@st.cache
def get_unique(data_series):
    data_u, data_c = np.unique(data_series.dropna(), return_counts=True)
    six = np.argsort(data_c)[::-1]
    data_u = data_u[six]
    data_c = data_c[six]

    return data_u, data_c

@st.cache
def transform_data(df):
    all_cols = pd.read_csv('../models/RandomForest/rf_cv_model_train_dataset_200629.csv', nrows=1, header=None).values[0]
    cols2use = ['advertiser', 'model', 'brand', 'power', 'mileage', 'first_registration_year', 'city', 'fuel', 'cylinder', 'gear_type']
    df1 = pd.get_dummies(df[cols2use], prefix=['advertiser', 'model', 'brand', 'city', 'fuel', 'gear_type'], prefix_sep=':')    
    dfX = pd.DataFrame(columns=all_cols)
    dfX = dfX.append(df1, ignore_index=True).fillna(0)
    dfX = dfX[all_cols]

    return dfX

@st.cache
def predict_price(df, clf):
    dfX = transform_data(df)
    y_pred = clf.predict(dfX)

    return y_pred



df = load_dataset(filepath)
clf = load_model(modelpath)


# Global counts
advertiser_u, advertiser_c = get_unique(df['advertiser'])
brands_u, brands_c = get_unique(df['brand'])
model_u, model_c = get_unique(df['model'])
city_u, city_c = get_unique(df['city'])
color_u, color_c = get_unique(df['color'])
cylinder_u, cylinder_c = get_unique(df['cylinder'])
first_registration_year_u, first_registration_year_c = get_unique(df['first_registration_year'])
fuel_u, fuel_c = get_unique(df['fuel'])
gear_type_u, gear_type_c = get_unique(df['gear_type'])
mileage_u, mileage_c = get_unique(df['mileage'])
power_u, power_c = get_unique(df['power'])
price_u, price_c = get_unique(df['price'])
segment_u, segment_c = get_unique(df['segment'])
version_u, version_c = get_unique(df['version'])


cols2use = ['advertiser', 'model', 'brand', 'power', 'mileage', 'first_registration_year', 'city', 'fuel', 'cylinder', 'gear_type']

advertiser_select = st.selectbox("Advertiser", advertiser_u)
model_select = st.selectbox("model", model_u)
brand_select = st.selectbox("brand", brands_u)
power_select = st.number_input("power", 0, 1300, 100, 1)
mileage_select = st.number_input("mileage", 0, 500000, 100000, 1000)
first_registration_year_select = st.selectbox("first_registration_year", first_registration_year_u)
city_select = st.selectbox("city", city_u)
fuel_select = st.selectbox("fuel", fuel_u)
cylinder_select = st.number_input("cylinder", 50, 6000, 100, 10)
gear_type_select = st.selectbox("gear_type", gear_type_u)

