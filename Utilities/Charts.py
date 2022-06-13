import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime as dt
import Utilities.Weather as uw

class Seas_Weather_Chart():
    """
    w_df_all: \n
        it MUST have only 1 weather variable, otherwise the sub doesn't know what to chart
    """
    def __init__(self, w_df_all, ext_mode=[], limit=[], cumulative = False, chart_df_ext = uw.WD_H_GFS, ref_year=uw.CUR_YEAR, ref_year_start = dt(uw.CUR_YEAR,1,1)):
        self.all_figs = {}
        self.w_df_all=w_df_all
        self.ext_mode=ext_mode
        self.limit=limit
        self.cumulative=cumulative
        self.chart_df_ext=chart_df_ext
        self.ref_year=ref_year
        self.ref_year_start=ref_year_start
        self.chart_all()

    def chart(self, w_df_all):
        cur_year_proj = str(uw.CUR_YEAR)+uw.PROJ
        w_var=w_df_all[uw.WD_HIST].columns[0].split('_')[1]

        has_fore = True
        if (w_var== uw.WV_HUMI) or (w_var== uw.WV_VVI) or (w_var== uw.WV_TEMP_SURF): has_fore=False
        df = uw.seasonalize(w_df_all[uw.WD_HIST], mode=self.ext_mode, limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
        
        if has_fore:           
            pivot_gfs = uw.seasonalize(w_df_all[uw.WD_GFS],mode=self.ext_mode,limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            fvi_fore_gfs = pivot_gfs.first_valid_index()
            
            pivot_ecmwf = uw.seasonalize(w_df_all[uw.WD_ECMWF],mode=self.ext_mode,limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            fvi_fore_ecmwf = pivot_ecmwf.first_valid_index()

            pivot_gfs = uw.seasonalize(w_df_all[uw.WD_H_GFS],mode=self.ext_mode,limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            pivot_ecmwf = uw.seasonalize(w_df_all[uw.WD_H_ECMWF],mode=self.ext_mode,limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
                            
        # Choose here what forecast to use to create the EXTENDED chart
        df_ext = uw.extend_with_seasonal_df(w_df_all[self.chart_df_ext], modes=self.ext_mode, limits=self.limit, ref_year=self.ref_year, ref_year_start=self.ref_year_start)

        # The below calculates the analog with current year already extended, so an analogue from 1/1 to 31/12 (that it is not useful)
        df_ext_pivot = uw.seasonalize(df_ext, mode=self.ext_mode,limit=self.limit,ref_year=self.ref_year,ref_year_start=self.ref_year_start)

        if self.cumulative:  
            df = uw.cumulate_seas(df, excluded_cols= ['Max','Min','Mean', cur_year_proj])
            df_ext_pivot = uw.cumulate_seas(df_ext_pivot,excluded_cols=['Max','Min','Mean'])
            pivot_gfs = uw.cumulate_seas(pivot_gfs,excluded_cols=['Max','Min','Mean'])
            pivot_ecmwf = uw.cumulate_seas(pivot_ecmwf,excluded_cols=['Max','Min','Mean'])

        fig = go.Figure()
        # Max - Min - Mean
        fig.add_trace(go.Scatter(x=df.index, y=df['Min'],fill=None,mode=None,line_color='lightgrey',name='Min',showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Max'],fill='tonexty',mode=None,line_color='lightgrey',name='Max',showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Mean'],mode='lines',line=dict(color='red',width=2), name='Mean',legendrank=uw.CUR_YEAR+2, showlegend=True))
        
        # Actuals
        for y in df.columns:       
            if ((y!='Max') and (y!='Min') and (y!='Mean') and (y!= cur_year_proj) and (uw.ANALOG not in str(y))):
                # Make the last 3 years visible
                if y>=uw.CUR_YEAR-3:
                    visible=True
                else: 
                    visible='legendonly'        

                # Use Black for the current year
                if y==uw.CUR_YEAR:
                    fig.add_trace(go.Scatter(x=df.index, y=df[y],mode='lines', legendrank=y, name=str(y),line=dict(color = 'black', width=2.5),visible=visible))
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df[y],mode='lines',legendrank=y, name=str(y),line=dict(width=1.5),visible=visible))
                    
        # Forecasts
        if has_fore:
            # GFS
            df_dummy=pivot_gfs[pivot_gfs.index>=fvi_fore_gfs]            
            fig.add_trace(go.Scatter(x=df_dummy.index,y=df_dummy[uw.CUR_YEAR],mode='lines+markers',line=dict(color='black',width=2,dash='dash'), name='GFS',legendrank=uw.CUR_YEAR+5, showlegend=True))
            
            # ECMWF
            df_dummy=pivot_ecmwf[pivot_ecmwf.index>=fvi_fore_ecmwf]            
            fig.add_trace(go.Scatter(x=df_dummy.index,y=df_dummy[uw.CUR_YEAR],mode='lines',line=dict(color='black',width=2,dash='dot'), name='ECMWF',legendrank=uw.CUR_YEAR+4, showlegend=True))
        
        
        # Analog Charting
        if (self.chart_df_ext == uw.WD_H_GFS):
            df_dummy=pivot_gfs
        elif (self.chart_df_ext == uw.WD_H_ECMWF):
            df_dummy=pivot_ecmwf

        analog_cols = [c for c in df_dummy.columns if uw.ANALOG in str(c)]
        for c in analog_cols:
            fig.add_trace(go.Scatter(x=df_dummy.index, y=df_dummy[c],mode='lines', name=c,legendrank=uw.CUR_YEAR+3,line=dict(color='green',width=1.5),visible=True))
        
        # Projection Charting
        df_dummy=df_ext_pivot
        fig.add_trace(go.Scatter(x=df_dummy.index, y=df_dummy[uw.CUR_YEAR],mode='lines',line=dict(color='darkred',width=2,dash='dash'), name=cur_year_proj,legendrank=uw.CUR_YEAR+1, showlegend=True))


        #region formatting
        fig.update_xaxes(tickformat="%d %b")
        # title={'text': w_df_all[uw.WD_HIST].columns[0],'font_size':15}
        # fig.update_layout(autosize=True,font=dict(size=12),title=title,hovermode="x unified",margin=dict(l=20, r=20, t=50, b=20))
        fig.update_layout(autosize=True,font=dict(size=12),hovermode="x unified",margin=dict(l=20, r=20, t=50, b=20))
        fig.update_layout(width=1400,height=787)

        # fig.update_layout(autosize=True,font=dict(size=10),title=title,margin=dict(l=20, r=20, t=50, b=20))
        # fig.show(renderer="browser")
        return fig
        #endregion

    def chart_all(self):        
        for col in self.w_df_all[uw.WD_HIST].columns:
            w_df_all={}

            for wd, w_df in self.w_df_all.items():                
                w_df_all[wd]=w_df[[col]]            

            self.all_figs[col]=self.chart(w_df_all)
            