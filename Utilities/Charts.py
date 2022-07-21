import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime as dt
import Utilities.Weather as uw
import Utilities.GLOBAL as GV

def find_on_x_axis(date, chart):
    id = 100*date.month+date.day
    for x in chart.data[0]['x']:
        if 100*x.month + x.day==id:
            return x

def add_interval_on_chart(chart, intervals=[], interval_index = GV.CUR_YEAR, text=[], position=['top left'],color=['red']):
    for i,d in enumerate(intervals):
        s=find_on_x_axis(d['start'][interval_index],chart)
        e=find_on_x_axis(d['end'][interval_index],chart)
                
        s_str=s.strftime("%Y-%m-%d")
        e_str=e.strftime("%Y-%m-%d")
        
        c= text[i] +'   ('+s.strftime("%b%d")+' - '+e.strftime("%b%d")+')'

        chart.add_vrect(x0=s_str, x1=e_str,fillcolor=color[i], opacity=0.1,layer="below", line_width=0, annotation=dict(font_size=14,textangle=90,font_color=color[i]), annotation_position=position[i], annotation_text=c)    

class Seas_Weather_Chart():
    """
    w_df_all: \n
        it MUST have only 1 weather variable, otherwise the sub doesn't know what to chart
    """
    def __init__(self, w_df_all, ext_mode=GV.EXT_DICT, cumulative = False, chart_df_ext = GV.WD_H_GFS, ref_year=GV.CUR_YEAR, ref_year_start = dt(GV.CUR_YEAR,1,1), hovermode='x unified'):
        self.all_figs = {}
        self.w_df_all=w_df_all
        self.ext_mode=ext_mode
        self.cumulative=cumulative
        self.chart_df_ext=chart_df_ext
        self.ref_year=ref_year
        self.ref_year_start=ref_year_start
        self.hovermode=hovermode

        self.chart_all()

    def chart(self, w_df_all):
        """
        the "seasonalize" function is called many times because it needs to put align everything for charting on the Last Leap Year time line
        """               
        cur_year_proj = str(GV.CUR_YEAR)+GV.PROJ
        w_var=w_df_all[GV.WD_HIST].columns[0].split('_')[1]

        has_fore = True
        if (w_var== GV.WV_HUMI) or (w_var== GV.WV_VVI) or (w_var==GV.WV_TEMP_SURF): has_fore=False
        # print(''); print(GV.WD_HIST);
        df = uw.seasonalize(w_df_all[GV.WD_HIST], mode=self.ext_mode, ref_year=self.ref_year,ref_year_start=self.ref_year_start)
        
        if has_fore:   
            # print(''); print(GV.WD_GFS);
            pivot_gfs = uw.seasonalize(w_df_all[GV.WD_GFS],mode=self.ext_mode,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            fvi_fore_gfs = pivot_gfs.first_valid_index()
            # print(''); print(GV.WD_ECMWF);
            pivot_ecmwf = uw.seasonalize(w_df_all[GV.WD_ECMWF],mode=self.ext_mode, ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            fvi_fore_ecmwf = pivot_ecmwf.first_valid_index()


            # print(''); print(GV.WD_H_GFS);
            pivot_h_gfs = uw.seasonalize(w_df_all[GV.WD_H_GFS],mode=self.ext_mode,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
            # print(''); print(GV.WD_H_ECMWF);
            pivot_h_ecmwf = uw.seasonalize(w_df_all[GV.WD_H_ECMWF],mode=self.ext_mode,ref_year=self.ref_year,ref_year_start=self.ref_year_start)
                            
        # Choose here what forecast to use to create the EXTENDED chart
        df_ext = uw.extend_with_seasonal_df(w_df_all[self.chart_df_ext], var_mode_dict=self.ext_mode, ref_year=self.ref_year, ref_year_start=self.ref_year_start)

        # The below calculates the analog with current year already extended
        # Using the analog from 1/1 to 31/12 is not right.
        # So, in the below section 'Analog Charting', to chart understand which analog to use there is an if statement to choose between:
        #       - pivot_h_gfs
        #       - pivot_h_ecmwf
        # Calculating 'df_ext_pivot' is done because it is what it is actually charted, but it cannot be used for picking the Analog

        df_ext_pivot = uw.seasonalize(df_ext, mode=self.ext_mode, ref_year=self.ref_year, ref_year_start=self.ref_year_start)

        if self.cumulative:  
            df = uw.cumulate_seas(df, excluded_cols= ['Max','Min','Mean', cur_year_proj])
            df_ext_pivot = uw.cumulate_seas(df_ext_pivot,excluded_cols=['Max','Min','Mean'])
            pivot_h_gfs = uw.cumulate_seas(pivot_h_gfs,excluded_cols=['Max','Min','Mean'])
            pivot_h_ecmwf = uw.cumulate_seas(pivot_h_ecmwf,excluded_cols=['Max','Min','Mean'])

        
        # Colors
        # https://plotly.com/python-api-reference/generated/plotly.express.colors.html
        # color_scale = px.colors.sequential.RdBu # https://plotly.com/python/builtin-colorscales/
        color_scale = px.colors.qualitative.Light24 # https://plotly.com/python/discrete-color/
        # print(px.colors.named_colorscales())
        
        i_color=1

        fig = go.Figure()
        # Max - Min - Mean
        fig.add_trace(go.Scatter(x=df.index, y=df['Min'],fill=None,mode=None,line_color='lightgrey',name='Min',showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Max'],fill='tonexty',mode=None,line_color='lightgrey',name='Max',showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Mean'],mode='lines',line=dict(color='blue',width=1.5), name='Mean',legendrank=GV.CUR_YEAR+2, showlegend=True))
        

        # Actuals
        for y in df.columns:       
            if ((y!='Max') and (y!='Min') and (y!='Mean') and (y!= cur_year_proj) and (GV.ANALOG not in str(y))):
                # Make the last 3 years visible
                if y>=GV.CUR_YEAR-0:
                    visible=True
                else: 
                    visible='legendonly'        

                # Use Black for the current year                

                if y==GV.CUR_YEAR:
                    fig.add_trace(go.Scatter(x=df.index, y=df[y],mode='lines', legendrank=y, name=str(y),line=dict(color = 'black', width=2.5),visible=visible))
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df[y],mode='lines',legendrank=y, name=str(y),line=dict(color = color_scale[i_color], width=1.5),visible=visible))

                i_color+=1
                if i_color==len(color_scale):i_color=0
                    
        # Forecasts
        if has_fore:
            # GFS
            df_dummy=pivot_h_gfs[pivot_h_gfs.index>=fvi_fore_gfs]            
            fig.add_trace(go.Scatter(x=df_dummy.index,y=df_dummy[GV.CUR_YEAR],mode='lines+markers',line=dict(color='black',width=2,dash='dash'), name='GFS',legendrank=GV.CUR_YEAR+5, showlegend=True))
            
            # ECMWF
            df_dummy=pivot_h_ecmwf[pivot_h_ecmwf.index>=fvi_fore_ecmwf]            
            fig.add_trace(go.Scatter(x=df_dummy.index,y=df_dummy[GV.CUR_YEAR],mode='lines',line=dict(color='black',width=2,dash='dot'), name='ECMWF',legendrank=GV.CUR_YEAR+4, showlegend=True))
        
        
        # Analog Charting        
        if (self.chart_df_ext == GV.WD_H_GFS):
            df_dummy=pivot_h_gfs
        elif (self.chart_df_ext == GV.WD_H_ECMWF):
            df_dummy=pivot_h_ecmwf

        analog_cols = [c for c in df_dummy.columns if GV.ANALOG in str(c)]

        for c in analog_cols:
            fig.add_trace(go.Scatter(x=df_dummy.index, y=df_dummy[c],mode='lines', name=c,legendrank=GV.CUR_YEAR+3,line=dict(color='green',width=1),visible=True))
        
        # Projection Charting
        lvi_hist=df[GV.CUR_YEAR].last_valid_index()
        df_dummy=df_ext_pivot[df_ext_pivot.index>=lvi_hist]
        fig.add_trace(go.Scatter(x=df_dummy.index, y=df_dummy[GV.CUR_YEAR],mode='lines',line=dict(color='red',width=2,dash='dash'), name=cur_year_proj,legendrank=GV.CUR_YEAR+1, showlegend=True))

        #formatting
        fig.update_xaxes(tickformat="%d %b")
        fig.update_layout(autosize=True,font=dict(size=12),hovermode=self.hovermode,margin=dict(l=20, r=20, t=50, b=20))
        fig.update_layout(width=1400,height=787)
        return fig

    def chart_all(self):        
        for col in self.w_df_all[GV.WD_HIST].columns:
            w_df_all={}

            for wd, w_df in self.w_df_all.items():                
                w_df_all[wd]=w_df[[col]]            

            self.all_figs[col]=self.chart(w_df_all)


def add_series(fig,x,y,name=None,mode='lines+markers',showlegend=True,line_width=1.0,color='black',marker_size=5,legendrank=0):
    fig.add_trace(go.Scatter(x=x, y=y,mode=mode, line=dict(width=line_width,color=color), marker=dict(size=marker_size), name=name, showlegend=showlegend, legendrank=legendrank))

def line_chart(x,y,name=None,mode='lines+markers',showlegend=True,line_width=1.0,color='black',marker_size=5,legendrank=0,width=1400,height=600):
    fig = go.Figure()
    add_series(fig,x,y,name,mode=mode,showlegend=showlegend,line_width=line_width,color=color,marker_size=marker_size,legendrank=legendrank)
    update_layout(fig,marker_size,line_width,width,height)
    return fig  

def update_layout(fig,marker_size,line_width,width,height):
    fig.update_traces(marker=dict(size=marker_size),line=dict(width=line_width))
    fig.update_xaxes(tickformat="%d %b")
    fig.update_layout(autosize=True,font=dict(size=12),hovermode="x unified",margin=dict(l=20, r=20, t=50, b=20))
    fig.update_layout(width=width,height=height)


def add_bar(fig,x,y,name=None,color='black'):
    fig.add_trace(go.Bar(x=x,y=y,name=name,marker_color=color))


def bar_chart(x,y,name=None,color='black',width=1400,height=600):
    fig = go.Figure()    
    fig.add_trace(go.Bar(x=x,y=y,name=name,marker_color=color))
    fig.update_layout(width=width,height=height)
    return fig  

