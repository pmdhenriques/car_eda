import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib as jl
import datetime
from glob import glob
import os

year_now = int(datetime.date.today().strftime("%Y"))

@st.cache
def load_dataset(filepath):
    df = pd.read_csv(filepath)

    return df

def load_model():
    clf = jl.load('../models/RandomForest/rf_cv_model_200629.joblib')

    return clf

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
def get_unique(data_series):
    data_u, data_c = np.unique(data_series.dropna(), return_counts=True)
    six = np.argsort(data_c)[::-1]
    data_u = data_u[six]
    data_c = data_c[six]

    return data_u, data_c

@st.cache
def filter_dataset(filter_dict):
    df_f = pd.DataFrame(df)
    for col in filter_dict.keys():
        ix = []
        if (col == 'price') | (col == 'first_registration_year') | (col == 'mileage') | (col == 'power') | (col == 'cylinder'):
            ixx = np.where((df_f[col] >= filter_dict[col][0]) & (df_f[col] <= filter_dict[col][1]))[0].tolist()
            ix.append(ixx)
        else:
            for flt in filter_dict[col]:
                ixx = np.where(df_f[col] == flt)[0].tolist()
                ix.append(ixx)
        if any(ix) > 0:
            ix = np.concatenate(ix)
            df_f = df_f.loc[ix].reset_index(drop=True)

    return df_f

@st.cache
def get_uniques(filter_dict):
    df_f = filter_dataset(filter_dict)
    data_u_dict = {x:[] for x in filter_dict.keys()}
    for col in df_f.columns:
        data_u, data_c = get_unique(df_f[col])
        data_u_dict[col] = [data_u, data_c]

    return data_u_dict

@st.cache
def predict_price(df, clf):
    dfX = transform_data(df)
    y_pred = clf.predict(dfX)

    return y_pred

"""
# StandVirtual EDA
"""

datadir = '../data/'
datasets = glob(os.path.join(datadir, 'standvirtual_dataset_[0-9]*.csv'))
filepath = st.selectbox('Select dataset', datasets, len(datasets)-1)
df = load_dataset(filepath)

clf = load_model()


filter_dict = {x:[] for x in df.columns}

# Global counts
brands_u, brands_c = get_unique(df['brand'])
model_u, model_c = get_unique(df['model'])
advertiser_u, advertiser_c = get_unique(df['advertiser'])
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


## Sidebar

# Sliders

st.sidebar.title("Select your filters")

min_price, max_price = st.sidebar.slider("Price range", 0, 100000, [1000, 50000], 500)
filter_dict['price'] = [min_price, max_price]

min_year, max_year = st.sidebar.slider("Year range", 1990, year_now, [2000, year_now], 1)
filter_dict['first_registration_year'] = [min_year, max_year]

min_mileage, max_mileage = st.sidebar.slider("Mileage", 0, 400000, [0, 200000], 1000)
filter_dict['mileage'] = [min_mileage, max_mileage]

min_power, max_power = st.sidebar.slider("Power", 0, 1300, [0, 1300], 10)
filter_dict['power'] = [min_power, max_power]

min_cylinder, max_cylinder = st.sidebar.slider("Cylinder", 0, 9000, [0, 2000], 100)
filter_dict['cylinder'] = [min_cylinder, max_cylinder]

# Multi options

uniques = get_uniques(filter_dict)
segment_options = ['%s [%d]'%(x, uniques['segment'][1][i]) for i, x in enumerate(uniques['segment'][0])]
segment_select = st.sidebar.multiselect("Segment", segment_options)
filter_dict['segment'] = [x.split(' [')[0] for x in segment_select]

uniques = get_uniques(filter_dict)
brand_options = ['%s [%d]'%(x, uniques['brand'][1][i]) for i, x in enumerate(uniques['brand'][0])]
brand_select = st.sidebar.multiselect("Brand", brand_options)
filter_dict['brand'] = [x.split(' [')[0] for x in brand_select]

uniques = get_uniques(filter_dict)
model_options = ['%s [%d]'%(x, uniques['model'][1][i]) for i, x in enumerate(uniques['model'][0])]
model_select = st.sidebar.multiselect("Model", model_options)
filter_dict['model'] = [x.split(' [')[0] for x in model_select]

uniques = get_uniques(filter_dict)
version_options = ['%s [%d]'%(x, uniques['version'][1][i]) for i, x in enumerate(uniques['version'][0])]
version_select = st.sidebar.multiselect("Version", version_options)
filter_dict['version'] = [x.split(' [')[0] for x in version_select]

uniques = get_uniques(filter_dict)
fuel_options = ['%s [%d]'%(x, uniques['fuel'][1][i]) for i, x in enumerate(uniques['fuel'][0])]
fuel_select = st.sidebar.multiselect("Fuel", fuel_options)
filter_dict['fuel'] = [x.split(' [')[0] for x in fuel_select]

uniques = get_uniques(filter_dict)
color_options = ['%s [%d]'%(x, uniques['color'][1][i]) for i, x in enumerate(uniques['color'][0])]
color_select = st.sidebar.multiselect("Color", color_options)
filter_dict['color'] = [x.split(' [')[0] for x in color_select]

uniques = get_uniques(filter_dict)
region_options = ['%s [%d]'%(x, uniques['region'][1][i]) for i, x in enumerate(uniques['region'][0])]
region_select = st.sidebar.multiselect("Region", region_options)
filter_dict['region'] = [x.split(' [')[0] for x in region_select]

uniques = get_uniques(filter_dict)
advertiser_options = ['%s [%d]'%(x, uniques['advertiser'][1][i]) for i, x in enumerate(uniques['advertiser'][0])]
advertiser_select = st.sidebar.multiselect("Advertiser", advertiser_options)
filter_dict['advertiser'] = [x.split(' [')[0] for x in advertiser_select]

uniques = get_uniques(filter_dict)
doors_n_options = ['%s [%d]'%(x, uniques['doors_n'][1][i]) for i, x in enumerate(uniques['doors_n'][0])]
doors_n_select = st.sidebar.multiselect("Doors", doors_n_options)
filter_dict['doors_n'] = [x.split(' [')[0] for x in doors_n_select]

uniques = get_uniques(filter_dict)
gear_type_options = ['%s [%d]'%(x, uniques['gear_type'][1][i]) for i, x in enumerate(uniques['gear_type'][0])]
gear_type_select = st.sidebar.multiselect("Gear type", gear_type_options)
filter_dict['gear_type'] = [x.split(' [')[0] for x in gear_type_select]

uniques = get_uniques(filter_dict)
traction_options = ['%s [%d]'%(x, uniques['traction'][1][i]) for i, x in enumerate(uniques['traction'][0])]
traction_select = st.sidebar.multiselect("Traction", traction_options)
filter_dict['traction'] = [x.split(' [')[0] for x in traction_select]


## Bools

uniques = get_uniques(filter_dict)
accepts_recovery_options = ['%s [%d]'%(x, uniques['accepts_recovery'][1][i]) for i, x in enumerate(uniques['accepts_recovery'][0])]
accepts_recovery_select = st.sidebar.checkbox("Accepts recovery")
filter_dict['accepts_recovery'] = [int(accepts_recovery_select)]

uniques = get_uniques(filter_dict)
metallic_options = ['%s [%d]'%(x, uniques['metallic'][1][i]) for i, x in enumerate(uniques['metallic'][0])]
metallic_select = st.sidebar.checkbox("Metallic")
filter_dict['metallic'] = [int(metallic_select)]

uniques = get_uniques(filter_dict)
negociable_options = ['%s [%d]'%(x, uniques['negociable'][1][i]) for i, x in enumerate(uniques['negociable'][0])]
negociable_select = st.sidebar.checkbox("Negociable")
filter_dict['negociable'] = [int(negociable_select)]

uniques = get_uniques(filter_dict)
damaged_options = ['%s [%d]'%(x, uniques['damaged'][1][i]) for i, x in enumerate(uniques['damaged'][0])]
damaged_select = st.sidebar.checkbox("Damaged")
filter_dict['damaged'] = [int(damaged_select)]


df_f = filter_dataset(filter_dict)


#fig, ax = plt.subplots()
#sel_ix = brands_c > 100
#ax.barh(brands_u[sel_ix][::-1], brands_c[sel_ix][::-1])
#ax.set_xlabel('Count')
#ax.set_ylabel('Car brand')
#plt.tight_layout()
#st.pyplot(fig)

"""
### Price analysis
"""

fig, ax = plt.subplots()
y0 = df['price'].dropna().values
y0_stats = np.percentile(y0, [25,50,75])
y1 = df_f['price'].dropna().values
y1_stats = np.percentile(y1, [25,50,75])
bins = np.arange(0, 80000, 1000)
plt.hist(y0, bins=bins, density=True, label='all: %d [%d, %d]'%(y0_stats[0], y0_stats[1], y0_stats[2]), alpha=0.5, color='b')
plt.hist(y1, bins=bins, density=True, label='filtered: %d [%d, %d]'%(y1_stats[0], y1_stats[1], y1_stats[2]), alpha=0.5, color='r')
ylims = plt.ylim()
for i in range(len(y0_stats)):
    plt.plot([y0_stats[i],y0_stats[i]], ylims, ':b')
    plt.plot([y1_stats[i],y1_stats[i]], ylims, ':r')
plt.xlabel('Price (€)')
plt.ylabel('pdf')
plt.legend()
plt.tight_layout()
st.pyplot(fig)

fig2, ax2 = plt.subplots(2, 2, figsize=[10, 10])

ax2[0,0].hist(df['first_registration_year'], bins=np.arange(1990,year_now+1,1), density=True, label='all', alpha=0.5, color='b')
ax2[0,0].hist(df_f['first_registration_year'], bins=np.arange(1990,year_now+1,1), density=True, label='filtered', alpha=0.5, color='r')
ax2[0,0].set_xlabel('Registration year')
ax2[0,0].get_xticklabels()
ax2[0,0].tick_params(axis='x', labelrotation=45)
ax2[0,0].legend()

ax2[0,1].hist(df['mileage'], bins=np.arange(0,500000,5000), density=True, label='all', alpha=0.5, color='b')
ax2[0,1].hist(df_f['mileage'], bins=np.arange(0,500000,5000), density=True, label='filtered', alpha=0.5, color='r')
ax2[0,1].set_xlabel('Mileage (km)')
ax2[0,1].tick_params(axis='x', labelrotation=45)

ax2[1,0].hist(df['power'], bins=np.arange(0,600, 10), density=True, label='all', alpha=0.5, color='b')
ax2[1,0].hist(df_f['power'], bins=np.arange(0,600, 10), density=True, label='filtered', alpha=0.5, color='r')
ax2[1,0].set_xlabel('Power (cv)')
ax2[1,0].tick_params(axis='x', labelrotation=45)

ax2[1,1].hist(df['cylinder'], bins=np.arange(0,6000, 200), density=True, label='all', alpha=0.5, color='b')
ax2[1,1].hist(df_f['cylinder'], bins=np.arange(0,6000, 200), density=True, label='filtered', alpha=0.5, color='r')
ax2[1,1].set_xlabel('Cylinder (cm3)')
ax2[1,1].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
st.pyplot(fig2)

"""
### Price prediction
"""

use_rel = st.checkbox('Relative')

price_pred = np.round(predict_price(df_f, clf))
price_true = df_f['price'].values
price_diff = price_true-price_pred
price_diff_rel = price_diff/price_true

if use_rel:
    price_diff_six = np.argsort(price_diff_rel)
else:
    price_diff_six = np.argsort(price_diff)

df_p = df_f.loc[price_diff_six, ['brand','model','version','mileage','cylinder','power','first_registration_year','price']].reset_index(drop=True)
df_p['price_est'] = price_pred[price_diff_six]
df_p['price_diff'] = price_diff[price_diff_six]
df_p['price_diff_rel (%)'] = np.round(price_diff_rel[price_diff_six]*100)
df_p['link'] = df_f.loc[price_diff_six,'link'].values

df_p

"""
### Price time valuation
"""

