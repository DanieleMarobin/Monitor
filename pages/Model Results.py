# imports
import pandas as pd
import time
import numpy as np
import statsmodels.api as sm

import plotly.express as px


import APIs.QuickStats as qs

import Utilities.SnD as us
import Utilities.Weather as uw
import Utilities.Modeling as um
import Utilities.Charts as uc
import Utilities.Utilities as uu

from datetime import datetime as dt
import streamlit as st


uu.initialize()

st.set_page_config(page_title="Model Results",layout="wide",initial_sidebar_state="expanded")


# Title, Settings, Recalculate Button etc
st.markdown("# Model Results")


st.sidebar.markdown("# Model Calculation Settings")

# st.sidebar.markdown("#### Yield Analysis Start")
yield_analysis_start = st.sidebar.date_input("Yield Analysis Start", dt.today()+pd.DateOffset(-1))

calc_again = st.sidebar.button('Re-Calculate')

if calc_again:
    st.session_state['recalculate'] = True



# declarations
corn_states=['IA','IL','IN','OH','MO','MN','SD','NE']
years=range(1985,2023)

days=[]
yields=[]

progress_str_empty = st.empty()
progress_empty = st.empty()

metric_empty = st.empty()
chart_empty = st.empty()
line_empty = st.empty()
daily_input_empty= st.empty()
dataframe_empty = st.empty()


# CORE calculation
if st.session_state['recalculate']:        

    # getting the data and building weighted DF
    progress_empty.progress(0)

    # Yield    
    progress_str_empty.write('Getting the Yields...'); progress_empty.progress(0.1)
    df_yield=qs.get_QS_yields(years=years,cols_subset=['year','Value'])
    M_yield=df_yield.set_index('year')

    # Progress
    progress_str_empty.write('Planting Progress...'); progress_empty.progress(0.3)
    planting_df=qs.get_QS_progress(progress_var='planting', years=years, cols_subset=['week_ending','Value'])
    progress_str_empty.write('Silking Progress...'); progress_empty.progress(0.5)
    silking_df=qs.get_QS_progress(progress_var='silking',years=years,cols_subset=['week_ending','Value'])

    # Progress as of 15th May (Yifu calls it "Planting Date" in his file)
    M_plant_on_May15=us.progress_from_date(planting_df, sel_date='2021-05-15')

    # Select the Weather Stations
    progress_str_empty.write('Getting the weather Data...'); progress_empty.progress(0.7)
    df_w_sel = uw.get_w_sel_df()
    df_w_sel = df_w_sel[df_w_sel[uw.WS_COUNTRY_ALPHA] == 'USA']

    # Build the Weather DF
    w_vars = [uw.WV_PREC, uw.WV_TEMP_MAX, uw.WV_SDD_30]
    in_files = uw.WS_UNIT_ALPHA
    out_cols = uw.WS_UNIT_ALPHA
    w_df_all = uw.build_w_df_all(df_w_sel, w_vars, in_files, out_cols)

    # Build the Weights
    progress_str_empty.write('Getting the production from USDA...'); progress_empty.progress(0.9)
    weights = us.get_USA_prod_weights('CORN', 'STATE', years, corn_states); progress_empty.progress(1.0)

    # Weighted DataFrame
    w_w_df_all = uw.weighted_w_df_all(w_df_all, weights, output_column='USA')

    # Dates (calculated here so that it is easier to build scenarios later)
    plant_80_pct=us.dates_from_progress(planting_df, sel_percentage=80)

    # Planting dates are 80% planted -40 and +25 days
    start=plant_80_pct['date']+pd.DateOffset(-40)
    end=plant_80_pct['date']+pd.DateOffset(25)
    planting_dates=pd.DataFrame({'start':start,'end':end})

    # DATES: jul_aug are 80% planted +26 and +105 days
    start=plant_80_pct['date']+pd.DateOffset(26)
    end=plant_80_pct['date']+pd.DateOffset(105)
    jul_aug_dates=pd.DataFrame({'start':start,'end':end})

    # DATES: Pollination SDD (Dates are silking 50% -15 and +15 days)
    silk_50_pct=us.dates_from_progress(silking_df, sel_percentage=50)
    silk_50_pct_CUR_YEAR=pd.Series([dt(uw.CUR_YEAR,d.month,d.day) for d in silk_50_pct['date']]) # Adding current estimate for silking dates
    silk_50_pct.loc[uw.CUR_YEAR]= np.mean(silk_50_pct_CUR_YEAR)

    start=silk_50_pct['date']+pd.DateOffset(-15)
    end=silk_50_pct['date']+pd.DateOffset(15)
    pollination_dates=pd.DataFrame({'start':start,'end':end})

    # DATES: Regular SDD (Dates are 20 Jun - 15 Sep)
    start=[dt(y,6,20) for y in silk_50_pct.index]
    end=[dt(y,9,25) for y in silk_50_pct.index]
    regular_dates=pd.DataFrame({'start':start,'end':end},index=silk_50_pct.index)

    # Saving the Dates
    st.session_state['dates']['plant_80'] = plant_80_pct
    st.session_state['dates']['silk_50'] = silk_50_pct

    st.session_state['dates']['planting'] = planting_dates
    st.session_state['dates']['jul_aug'] = jul_aug_dates
    st.session_state['dates']['pollination'] = pollination_dates
    st.session_state['dates']['regular'] = regular_dates
    

    # ------------------------------------- EXTEND THE WEATHER -------------------------------------
    # 
    # select which dataframe to extend
    df_to_ext =  w_w_df_all[uw.WD_H_GFS] # Extending with GFS
    # df_to_ext =  w_w_df_all[uw.WD_H_ECMWF] # Extending with ECMWF

    # Add the SDD
    # uw.add_Sdd(df_to_ext, source_WV=uw.WV_TEMP_MAX, threshold=30)
    # seas = seasonalize(w_df, col, mode=mode, limit=limit, ref_year=ref_year, ref_year_start=ref_year_start)
    w_w_df_ext, dict_col_seas = uw.extend_with_seasonal_df(df_to_ext)
    # ----------------------------------------------------------------------------------------------

    # build the final Model DataFrame

    # Copying to simple "w_df"
    w_df = w_w_df_ext.copy()

    # -------------------------------- 9 Variables --------------------------------
    # Trend                                                                             # 1
    # M_plant_on_May15                                                                  # 2
    M_jul_aug_prec = uw.extract_w_windows(w_df[['USA_Prec']],jul_aug_dates)             # 3
    # M_jul_aug_prec SQ                                                                 # 4
    M_planting_prec = uw.extract_w_windows(w_df[['USA_Prec']],planting_dates)           # 5
    # M_planting_prec                                                                   # 6
    M_pollination_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], pollination_dates)    # 7
    M_regular_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], regular_dates)            # 8
    # Precip_Interaction                                                                # 9


    # Combining the 2 SDD columns
    M_sdd = pd.concat([M_pollination_sdd, M_regular_sdd],axis=1)
    M_sdd.columns=['Pollination_SDD','Regular_SDD']
    M_sdd['Regular_SDD']=M_sdd['Regular_SDD']-M_sdd['Pollination_SDD']


    cols_names = ['Yield','Plant_Progr_May15','Jul_Aug_Prec','Pollination_SDD','Regular_SDD', 'Planting_Prec']

    M_df=[M_yield, M_plant_on_May15, M_jul_aug_prec/25.4, M_sdd*9/5, M_planting_prec/25.4]

    M_df=pd.concat(M_df,axis=1)
    M_df.columns=cols_names

    M_df['Trend']=M_df.index

    M_df['Jul_Aug_Prec_Sq']=M_df['Jul_Aug_Prec']**2 # Sq
    M_df['Planting_Prec_Sq']=M_df['Planting_Prec']**2 # Sq
    M_df['Precip_Interaction']=M_df['Planting_Prec']*M_df['Jul_Aug_Prec']

    y_col='Yield'
    df=M_df.dropna()

    y_df = df[[y_col]]
    X_df=df.drop(columns = y_col)
    
    X2_df = sm.add_constant(X_df)
    
    stats_model = sm.OLS(y_df, X2_df).fit()
    #

    # Iterations    
    # Initializing
    # Copy the last row (already initialized for everything apart for the things that need changing) 
    X_pred=M_df.loc[uw.CUR_YEAR:uw.CUR_YEAR].reset_index()    
    
    last_day = w_w_df_all[uw.WD_H_GFS].index[-1]    

    # only the last year is going to change
    for day in pd.date_range(yield_analysis_start, last_day):
        days.append(day)
        yields.append(np.NaN)
        daily_inputs={}
        
    dict_col_seas={}
    for i, day in enumerate(pd.date_range(yield_analysis_start, last_day)):    
        start = time.time()
        w_w_df_ext = uw.extend_with_seasonal_df(df_to_ext.loc[:day],dict_col_seas=dict_col_seas)[0]
        
        # build the final Model DataFrame

        # Copying to simple "w_df"
        w_df = w_w_df_ext.copy()
        
        # -------------------------------- Weather Related Variables --------------------------------
        M_jul_aug_prec = uw.extract_w_windows(w_df[['USA_Prec']],jul_aug_dates.loc[uw.CUR_YEAR:uw.CUR_YEAR])
        M_planting_prec = uw.extract_w_windows(w_df[['USA_Prec']],planting_dates.loc[uw.CUR_YEAR:uw.CUR_YEAR])
        M_pollination_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], pollination_dates.loc[uw.CUR_YEAR:uw.CUR_YEAR])
        M_regular_sdd = uw.extract_w_windows(w_df[['USA_Sdd30']], regular_dates.loc[uw.CUR_YEAR:uw.CUR_YEAR])


        # -------------------------------- 9 Variables --------------------------------
        X_pred.loc[i,:]=M_df.loc[uw.CUR_YEAR]
        # Trend                                                         # 1
        # M_plant_on_May15                                              # 2
        X_pred.loc[i,'Jul_Aug_Prec']=M_jul_aug_prec.values[0]/25.4           # 3
        X_pred.loc[i,'Jul_Aug_Prec_Sq']=(M_jul_aug_prec.values[0]/25.4)**2     # 4
        X_pred.loc[i,'Planting_Prec']=M_planting_prec.values[0]/25.4         # 5
        X_pred.loc[i,'Planting_Prec_Sq']=(M_planting_prec.values[0]/25.4)**2   # 6
        X_pred.loc[i,'Pollination_SDD']=M_pollination_sdd.values[0]*9/5     # 7
        X_pred.loc[i,'Regular_SDD']=(M_regular_sdd.values[0]- M_pollination_sdd.values[0])*9/5            # 8
        X_pred.loc[i,'Precip_Interaction']=X_pred.loc[i,'Planting_Prec']*X_pred.loc[i,'Jul_Aug_Prec']# 9        


    X_pred_2 = sm.add_constant(X_pred, has_constant='add')
    yields = stats_model.predict(X_pred_2[stats_model.params.index])

    # Save Iteration Info
    daily_inputs=X_pred.drop(columns=['index'])
    daily_inputs['Yield']=yields
    daily_inputs['day']=[d.strftime("%d %b %Y") for d in days]
    daily_inputs=daily_inputs.set_index('day')
        
    st.session_state['daily_inputs']=daily_inputs
    st.session_state['final_df'] = df.copy()
    st.session_state['yields'] = yields.values.copy()
    st.session_state['days'] = days.copy()
    st.session_state['model'] = stats_model

    # metric_empty.metric(label='Yield - '+days[-1].strftime("%d %b %Y"), value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")
    chart_empty.plotly_chart(uc.line_chart(x=days,y=yields))
    line_empty.markdown('---')
    daily_input_empty.markdown('##### Daily Inputs')
    dataframe_empty.dataframe(daily_inputs)

    progress_str_empty.success('All Done!')
    st.session_state['recalculate'] = False
# copy variables (in case we don't need to calculate the model again)
else:
    
    # Assign the saved values to the variables
    
    df = st.session_state['final_df']    
    days = st.session_state['days']    
    yields = st.session_state['yields']    
    stats_model = st.session_state['model']

    metric_empty.metric(label='Yield', value="{:.2f}".format(yields[-1]), delta= "{:.2f}".format(yields[-1]-yields[-2])+" bu/Ac")    
    chart_empty.plotly_chart(uc.line_chart(x=days,y=yields))    
    line_empty.markdown('---')        
    daily_input_empty.markdown('##### Daily Inputs')    
    dataframe_empty.dataframe(st.session_state['daily_inputs'])    
    

# -------------------------------------------- Model Details --------------------------------------------
# coefficients
model_coeff=pd.DataFrame(columns=stats_model.params.index)
model_coeff.loc[len(model_coeff)]=stats_model.params.values
model_coeff=model_coeff.drop(columns=['const'])
model_coeff.index=['Model Coefficients']

st.markdown('##### Coefficients')
st.dataframe(model_coeff)


# Dates
dates_fmt = "%d %b %Y"

st.markdown('---')
st.markdown('### Key Dates')

col_plant, col_jul_aug, col_regular, col_pollination = st.columns([1, 1,1,1])

with col_plant:
    st.markdown('##### Planting_Prec')
    st.write('80% planted -40 and +25 days')
    styler = st.session_state['dates']['planting'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_jul_aug:
    st.markdown('##### Jul_Aug_Prec')    
    st.write('80% planted +26 and +105 days')
    styler = st.session_state['dates']['jul_aug'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_regular:
    st.markdown('##### Regular_SDD')
    st.write('20 Jun - 15 Sep')
    styler = st.session_state['dates']['regular'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_pollination:
    # 50% Silking -15 and +15 days
    st.markdown('##### Pollination_SDD')
    st.write('50% Silking -15 and +15 days')
    styler = st.session_state['dates']['pollination'].sort_index(ascending=False).style.format({"start": lambda t: t.strftime(dates_fmt),"end": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

#

# Key Milestones
st.markdown('---')
st.markdown('### Key Progress Milestones')
col_plant_80, col_silk_50, d_0,d_1 = st.columns([1, 1,1,1])

with col_plant_80:
    st.markdown('##### 80% Planted')    
    styler = st.session_state['dates']['plant_80'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)

with col_silk_50:
    st.markdown('##### 50% Silking')
    styler = st.session_state['dates']['silk_50'].sort_index(ascending=False).style.format({"date": lambda t: t.strftime(dates_fmt)})
    st.write(styler)      


# final DataFrame
st.markdown('---')
st.markdown('### Final DataFrame')
st.dataframe(df.sort_index(ascending=False))


# summary
st.markdown("---")
st.subheader('Model Summary:')
# st.write(stats_model.summary())

st.write(stats_model.summary())


# Correlation Matrix
st.markdown("---")
st.subheader('Correlation Matrix:')
st.plotly_chart(um.chart_corr_matrix(df.drop(columns=['Yield'])))
