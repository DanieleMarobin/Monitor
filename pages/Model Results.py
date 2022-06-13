import pandas as pd
import numpy as np
import statsmodels.api as sm

import plotly.express as px


import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um

from datetime import datetime as dt
import streamlit as st

st.set_page_config(page_title="Model Results",layout="wide",initial_sidebar_state="expanded")

col_model_text, col_calc_again = st.columns([3, 1])

with col_model_text:
    st.markdown("# Model Calculation and Results")

with col_calc_again:
    st.markdown("# ")
    calc_again = st.button('Re-Calculate')

# A button to decrement the counter

if calc_again:
    st.session_state['count'] = 0


st.markdown("---")
st.sidebar.markdown("# Model Calculation and Results")


corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
years=range(1985,2023)


# Yield     
st.write('Getting the Yields...')
df_yield=qs.get_QS_yields(years=years,cols_subset=['year','Value'])
M_yield=df_yield.set_index('year')


# Progress
st.write('Planting Progress...')
planting_df=qs.get_QS_progress(progress_var='planting', years=years, cols_subset=['week_ending','Value'])
st.write('Silking Progress...')
silking_df=qs.get_QS_progress(progress_var='silking',years=years,cols_subset=['week_ending','Value'])


# Progress as of 15th May (Yifu calls it "Planting Date" in his file)
M_plant_on_May15=us.progress_from_date(planting_df, sel_date='2021-05-15')


# Select the Weather Stations
st.write('Getting the weather Data...')
df_w_sel = uw.get_w_sel_df()
df_w_sel = df_w_sel[df_w_sel[uw.WS_COUNTRY_ALPHA] == 'USA']

# Build the Weather DF
w_vars = [uw.WV_PREC, uw.WV_TEMP_MAX]
in_files = uw.WS_UNIT_ALPHA
out_cols = uw.WS_UNIT_ALPHA
w_df_all = uw.build_w_df_all(df_w_sel, w_vars, in_files, out_cols)


# Build the Weights
st.write('Getting the production from USDA...')
weights = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states)

st.write('Building the Model DataFrame...')
# Weighted DataFrame
w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

# Weighted DataFrame
w_w_df_h_ecwmf_ext = uw.extend_with_seasonal_df(w_w_df_all[uw.WD_H_ECMWF], modes=[uw.EXT_MEAN])

# Copying to simple "w_df"
w_df = w_w_df_h_ecwmf_ext.copy()


# jul_aug dates are 80% planted +26 and +105 days
plant_80_pct=us.dates_from_progress(planting_df, sel_percentage=80)
start=plant_80_pct['date']+pd.DateOffset(26)
end=plant_80_pct['date']+pd.DateOffset(105)

# Dates
jul_aug_dates=pd.DataFrame({'start':start,'end':end})    
# Model column
M_jul_aug_prec = uw.extract_w_windows(w_df[['USA_Prec']],jul_aug_dates) 


# Planting dates are 80% planted -40 and +25 days
start=plant_80_pct['date']+pd.DateOffset(-40)
end=plant_80_pct['date']+pd.DateOffset(25)

# Dates
planting_dates=pd.DataFrame({'start':start,'end':end})

# Model column
M_planting_prec = uw.extract_w_windows(w_df[['USA_Prec']],planting_dates)


# Stress Degree Day (SDD)
sdd_df=w_df[['USA_TempMax']].copy()
mask=sdd_df.USA_TempMax>30.0
sdd_df[mask]=sdd_df[mask]-30
sdd_df[~mask]=0


# Pollination SDD (Dates are silking 50% -15 and +15 days)
silk_50_pct=us.dates_from_progress(silking_df, sel_percentage=50)

# Adding current estimate for silking dates
silk_50_pct_CUR_YEAR=pd.Series([dt(uw.CUR_YEAR,d.month,d.day) for d in silk_50_pct['date']])
silk_50_pct.loc[uw.CUR_YEAR]= np.mean(silk_50_pct_CUR_YEAR)

start=silk_50_pct['date']+pd.DateOffset(-15)
end=silk_50_pct['date']+pd.DateOffset(15)

# Dates
pollination_dates=pd.DataFrame({'start':start,'end':end})

# st.write(pollination_dates.sort_index(ascending=False))

# Model column
M_pollination_sdd = uw.extract_w_windows(sdd_df, pollination_dates)


# Regular SDD (Dates are 20 Jun - 15 Sep)
start=[dt(y,6,20) for y in silk_50_pct.index]
end=[dt(y,9,25) for y in silk_50_pct.index]

# Dates
regular_dates=pd.DataFrame({'start':start,'end':end},index=silk_50_pct.index)

# Model column
M_regular_sdd = uw.extract_w_windows(sdd_df, regular_dates)


# Combining the 2 SDD columns
M_sdd = pd.concat([M_pollination_sdd, M_regular_sdd],axis=1)
M_sdd.columns=['Pollination_SDD','Regular_SDD']
M_sdd['Regular_SDD']=M_sdd['Regular_SDD']-M_sdd['Pollination_SDD']



#region dates
dates_fmt = "%d %b %Y"
st.markdown('---')
st.markdown('### Key Progress Milestones')
col_plant_80, col_silk_50, d_0,d_1 = st.columns([1, 1,1,1])

with col_plant_80:
    st.markdown('##### 80% Planted')    
    styler = plant_80_pct.sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_silk_50:
    st.markdown('##### 50% Silking')
    styler = silk_50_pct.sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)    


st.markdown('---')
st.markdown('### Key Dates')

st.session_state['dates']['planting'] = planting_dates
st.session_state['dates']['jul_aug'] = jul_aug_dates
st.session_state['dates']['pollination'] = pollination_dates
st.session_state['dates']['regular'] = regular_dates    


col_plant, col_jul_aug, col_regular, col_pollination = st.columns([1, 1,1,1])

with col_plant:
    st.markdown('##### Planting_Prec')
    st.write('80% planted -40 and +25 days')
    styler = planting_dates.sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_jul_aug:
    st.markdown('##### Jul_Aug_Prec')    
    st.write('80% planted +26 and +105 days')
    styler = jul_aug_dates.sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_regular:
    st.markdown('##### Regular_SDD')
    st.write('20 Jun - 15 Sep')
    styler = regular_dates.sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_pollination:
    # 50% Silking -15 and +15 days
    st.markdown('##### Pollination_SDD')
    st.write('50% Silking -15 and +15 days')
    styler = pollination_dates.sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)
#endregion

#region build Model DataFrame
cols_names = ['Yield','Planting_Date','Jul_Aug_Prec','Pollination_SDD','Regular_SDD', 'Planting_Prec']

M_df=[M_yield, M_plant_on_May15, M_jul_aug_prec/25.4, M_sdd*9/5, M_planting_prec/25.4]

M_df=pd.concat(M_df,axis=1)
M_df.columns=cols_names


M_df['Trend']=M_df.index

M_df['Jul_Aug_Prec_Sq']=M_df['Jul_Aug_Prec']**2 # Sq
M_df['Planting_Prec_Sq']=M_df['Planting_Prec']**2 # Sq

M_df['Precip_Interaction']=M_df['Planting_Prec']*M_df['Jul_Aug_Prec']

st.markdown('---')
st.markdown('### Final DataFrame')

styler = M_df.sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
st.dataframe(styler)
#endregion

#region Fit the final Model
y_col='Yield'
df=M_df.dropna()

y_df = df[[y_col]]
X_df=df.drop(columns = y_col)

X2_df = sm.add_constant(X_df)    
stats_model = sm.OLS(y_df, X2_df).fit()
#endregion

#region Scenarios
df_2022=M_df.copy()
df_2022.loc[uw.CUR_YEAR,'Pollination_SDD']=df['Pollination_SDD'].mean()
df_2022.loc[uw.CUR_YEAR,'Regular_SDD']=df['Regular_SDD'].mean()
df_2022 = sm.add_constant(df_2022)
#endregion

#region Predicting
pred = stats_model.predict(df_2022[stats_model.params.index])[uw.CUR_YEAR]
#endregion

st.markdown("---")
st.markdown("### Prediction")
st.subheader(pred)
st.markdown("---")
st.subheader('Model Summary:')
st.write(stats_model.summary())

st.markdown("---")
st.subheader('Correlation Matrix:')
st.plotly_chart(um.chart_corr_matrix(X_df))

    


