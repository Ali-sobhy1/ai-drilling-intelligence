from __future__ import annotations
import io, os, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data' / 'drilling_demonstration_dataset.csv.gz'

st.set_page_config(page_title='AI Drilling Intelligence', page_icon='🛢️', layout='wide', initial_sidebar_state='expanded')

# ----------------------------- Theme / helpers -----------------------------
st.markdown('''<style>
:root{--bg:#07090d;--panel:#111722;--panel2:#151e2b;--line:#273447;--ink:#edf3fa;--muted:#93a4b8;--orange:#ff9d32;--cyan:#32c6df;--green:#43d6a0}
.stApp{background:radial-gradient(ellipse at 12% 0%,#182536 0%,#0b1018 38%,#07090d 78%);color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0e15,#0d1420);border-right:1px solid #263346}
[data-testid="stSidebar"] *{color:#dbe6f2}
.block-container{padding-top:1.15rem;padding-bottom:2.4rem;max-width:1680px}
[data-testid="stMetric"]{background:linear-gradient(145deg,#151e2a,#101620);border:1px solid #2a394c;padding:16px 18px;border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
[data-testid="stMetricLabel"]{color:#9fb1c5!important;font-size:.82rem;letter-spacing:.04em;text-transform:uppercase}
[data-testid="stMetricValue"]{color:#f4f7fb!important;font-weight:750}
[data-testid="stMetricDelta"]{font-weight:600}
div[data-testid="stExpander"]{border:1px solid #2a394c;border-radius:13px;background:rgba(17,23,34,.72)}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #a95f20;background:linear-gradient(135deg,#f28b26,#c96c18);color:#0a0e14;font-weight:750;min-height:2.55rem;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#ffc078;background:linear-gradient(135deg,#ffad4d,#e17b1e);color:#080b10;transform:translateY(-1px)}
.stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="select"]>div{background:#101824!important;color:#eaf2fb!important;border-color:#33445a!important;border-radius:9px!important}
.stTabs [data-baseweb="tab"]{color:#9eb0c4;font-weight:650}
.stTabs [aria-selected="true"]{color:#ffad4d!important;border-bottom-color:#ff9d32!important}
hr{border-color:#263548!important}
.small-note{color:#8fa2b7;font-size:.86rem}
.hero{position:relative;overflow:hidden;padding:25px 28px;border-radius:18px;background:linear-gradient(112deg,#111b29 0%,#152638 58%,#2a2118 100%);border:1px solid #34475d;margin-bottom:20px;box-shadow:0 12px 34px rgba(0,0,0,.22)}
.hero:after{content:"";position:absolute;right:-48px;top:-88px;width:250px;height:250px;border:1px solid rgba(255,157,50,.22);border-radius:50%;box-shadow:0 0 0 24px rgba(255,157,50,.035),0 0 0 50px rgba(50,198,223,.025)}
.hero h1{margin:0;color:#f7f9fc;font-size:2.05rem;font-weight:800;letter-spacing:-.025em}.hero p{margin:.55rem 0 0;color:#b4c3d3;font-size:.98rem;max-width:900px}
.tag{display:inline-block;border:1px solid #506176;background:#111d2a;color:#ffbd70;padding:5px 10px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.06em;margin:13px 6px 0 0}
.section-kicker{color:#ffad4d;font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}
[data-testid="stDataFrame"]{border:1px solid #2b3b50;border-radius:12px;overflow:hidden}
[data-testid="stPlotlyChart"]{background:rgba(13,19,29,.5);border:1px solid #202e40;border-radius:13px;padding:5px}
</style>''', unsafe_allow_html=True)

COL = {
 'date':'date','depth':'Hole depth (MD) m','block':'Block Position m','rop':'Rate of Penetration m/h',
 'rop5':'Rate of Penetration (5ft avg) m/h','rpm':'Average Rotary Speed rpm','bit_depth':'Bit Depth (MD) m',
 'hookload':'Average Hookload (Klb)','wob':'Weight on Bit (Klb)','torque':'Average Surface Torque (Klb.ft)',
 'spp':'Average Standpipe Pressure (psi)','flow':'Mud Flow In (gpm)'}
NUMERIC_KEYS=['depth','block','rop','rop5','rpm','bit_depth','hookload','wob','torque','spp','flow']
DISPLAY={k:v for k,v in COL.items() if k!='date'}

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><span class="tag">AI / ML</span><span class="tag">DRILLING ENGINEERING</span><span class="tag">DARK OPERATIONS UI</span><span class="tag">INTERACTIVE ANALYTICS</span></div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _read_default_dataset(path_text, modified_stamp):
    raw=pd.read_csv(path_text,compression='infer',low_memory=False)
    return _prepare_data(raw)

def _prepare_data(raw):
    raw.columns=[str(c).strip() for c in raw.columns]
    if COL['date'] in raw.columns:
        raw[COL['date']]=pd.to_datetime(raw[COL['date']],errors='coerce')
        raw=raw.sort_values(COL['date']).reset_index(drop=True)
    for k in NUMERIC_KEYS:
        if COL[k] in raw.columns: raw[COL[k]]=pd.to_numeric(raw[COL[k]],errors='coerce')
    return raw

def _safe_dataframe(frame, *args, **kwargs):
    """Normalize mixed object columns for reliable Arrow serialization in Streamlit."""
    if isinstance(frame, pd.DataFrame):
        frame = frame.copy()
        for col in frame.select_dtypes(include=['object']).columns:
            non_null_types = {type(v) for v in frame[col].dropna().head(2000)}
            if len(non_null_types) > 1:
                frame[col] = frame[col].map(lambda v: str(v) if pd.notna(v) else v)
    kwargs.setdefault('width', 'stretch')
    return st.dataframe(frame, *args, **kwargs)

def _chart_sample(frame, max_rows=8000):
    """Downsample only chart inputs; preserve the original dataset and chronological order."""
    if len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame)-1, max_rows, dtype=int)
    return frame.iloc[positions]

def load_data(uploaded=None):
    if uploaded is not None:
        if uploaded.name.lower().endswith('.csv') or uploaded.name.lower().endswith('.csv.gz'):
            raw=pd.read_csv(uploaded,compression='infer',low_memory=False)
        else: raw=pd.read_excel(uploaded)
        return _prepare_data(raw),f'Uploaded: {uploaded.name}'
    if DEFAULT_DATA.exists():
        raw=_read_default_dataset(str(DEFAULT_DATA),DEFAULT_DATA.stat().st_mtime)
        return raw,'Bundled demonstration dataset'
    return None,'No dataset loaded'

def has(df,key): return COL[key] in df.columns

def active_mask(df):
    m=pd.Series(True,index=df.index)
    if has(df,'rop'): m &= df[COL['rop']].fillna(0)>0
    if has(df,'wob'): m &= df[COL['wob']].fillna(0)>0
    if has(df,'rpm'): m &= df[COL['rpm']].fillna(0)>0
    return m

def numeric_features(df, keys):
    cols=[COL[k] for k in keys if has(df,k)]
    x=df[cols].replace([np.inf,-np.inf],np.nan)
    return x,cols

def chart_layout(fig, height=360):
    fig.update_layout(height=height,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=15,r=15,t=45,b=15),legend_title_text='')
    fig.update_xaxes(gridcolor='rgba(150,170,190,.12)',zerolinecolor='rgba(150,170,190,.15)')
    fig.update_yaxes(gridcolor='rgba(150,170,190,.12)',zerolinecolor='rgba(150,170,190,.15)')
    return fig

def add_time_features(df, target_col, feature_cols, lags=(1,2,3,5,10)):
    z=df.copy()
    out=pd.DataFrame(index=z.index)
    for c in feature_cols:
        for lag in lags:
            out[f'{c}_lag{lag}']=z[c].shift(lag)
    for w in (5,15):
        out[f'{target_col}_rollmean{w}']=z[target_col].shift(1).rolling(w,min_periods=w).mean()
        out[f'{target_col}_rollstd{w}']=z[target_col].shift(1).rolling(w,min_periods=w).std()
    return out

def train_rop_model(df, model_name, active_only=True):
    needed=['rop','wob','rpm','torque','hookload','spp','flow','depth']
    keys=[k for k in needed if has(df,k) and k!='rop']
    if not has(df,'rop') or len(keys)<3: raise ValueError('ROP and at least three predictor columns are required.')
    work=df[[COL['rop']]+[COL[k] for k in keys]].copy().replace([np.inf,-np.inf],np.nan)
    if active_only:
        mask=(work[COL['rop']]>0)
        if COL['wob'] in work: mask &= work[COL['wob']]>0
        if COL['rpm'] in work: mask &= work[COL['rpm']]>0
        work=work.loc[mask]
    work=work.dropna()
    if len(work)<100: raise ValueError('Not enough valid drilling observations for a reliable train/test evaluation.')
    X=work[[COL[k] for k in keys]]; y=work[COL['rop']]
    split=max(1,int(len(work)*.8)); Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
    if len(Xte)<10: raise ValueError('Not enough observations in chronological test partition.')
    models={
      'Random Forest':RandomForestRegressor(n_estimators=180,max_depth=14,min_samples_leaf=3,n_jobs=-1,random_state=42),
      'Extra Trees':ExtraTreesRegressor(n_estimators=180,max_depth=14,min_samples_leaf=3,n_jobs=-1,random_state=42),
      'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=180,learning_rate=.07,max_leaf_nodes=20,l2_regularization=1.0,random_state=42)}
    model=models[model_name]; model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={'R²':r2_score(yte,pred),'MAE':mean_absolute_error(yte,pred),'RMSE':mean_squared_error(yte,pred)**.5,'Train rows':len(Xtr),'Test rows':len(Xte)}
    return model,keys,metrics,pd.DataFrame({'Actual ROP':yte.values,'Predicted ROP':pred},index=yte.index),Xtr,Xte,ytr,yte

def get_model_importance(model, features, Xte=None, yte=None):
    if hasattr(model,'feature_importances_'):
        return pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
    if Xte is not None and yte is not None:
        pi=permutation_importance(model,Xte,yte,n_repeats=4,random_state=42,n_jobs=-1)
        return pd.DataFrame({'Feature':features,'Importance':pi.importances_mean}).sort_values('Importance',ascending=False)
    return pd.DataFrame()

def engineering_knowledge_answer(question: str, frame: pd.DataFrame) -> str:
    """Provide concise, rule-based drilling guidance without an external AI API."""
    q = question.lower()
    topics = [
        (('ecd', 'equivalent circulating density'),
         "ECD is the effective mud density while circulating. In field units, a common approximation is ECD (ppg) = MW (ppg) + annular pressure loss (psi) / (0.052 × TVD (ft)). Use TVD, not MD, in the hydrostatic conversion. You need mud weight, TVD and annular pressure loss at the point of interest. Validate rheology, geometry, temperature and pressure assumptions."),
        (('hole cleaning', 'cuttings transport'),
         "Hole cleaning depends on annular velocity, flow regime, fluid rheology, hole angle, cuttings size/shape, ROP, pipe rotation and eccentricity. Review flow rate, annular geometry, mud rheology, inclination and cuttings loading together. A surface dataset alone cannot confirm downhole bed height."),
        (('surge', 'swab'),
         "Surge and swab are transient pressure changes caused by pipe movement. Their magnitude depends on trip speed, annular geometry, mud rheology, clearance, BHA/tool joints and well geometry. A defensible calculation needs these inputs and a suitable hydraulics model; do not infer safe trip speed from surface trends alone."),
        (('torque and drag', 'torque', 'drag'),
         "Torque and drag are affected by well trajectory, friction, string/BHA geometry, contact forces, hole condition, cuttings beds and operating mode. Compare rotating/off-bottom torque, pick-up/slack-off hookload and trends by depth. Surface anomalies are screening indicators, not proof of a specific downhole mechanism."),
        (('stuck pipe', 'differential sticking', 'mechanical sticking'),
         "Stuck pipe mechanisms are commonly grouped into differential sticking and mechanical sticking (including pack-off/bridging, keyseating and wellbore instability). Diagnose using the event sequence, circulation response, overpull/slack-off, torque, mud properties, hole cleaning and wellbore conditions. Do not diagnose from one surface parameter alone."),
        (('well control', 'kick', 'shut in'),
         "For a suspected influx, follow the approved well-specific well-control procedures and the rig's shut-in policy immediately. Confirm flow with the designated indicators, communicate with the responsible supervisor and use the approved well-control method. This assistant is not a substitute for the well-control programme or certified training."),
        (('mse', 'mechanical specific energy'),
         "Mechanical Specific Energy (MSE) is an energy-per-volume indicator used to assess drilling efficiency. A common rotary-drilling form combines WOB/bit area and rotary torque × angular speed divided by bit area × ROP, with unit-consistent conversion factors. A valid calculation requires bit diameter and consistent units; MSE trends are not a standalone bit-dull or dysfunction diagnosis."),
        (('digital twin',),
         "A drilling Digital Twin combines a physical asset, synchronized data, a state/physics model, model updating, simulation and a human decision loop. A dashboard that predicts ROP from historical surface data is a data-driven surrogate or scenario demonstrator; it is not a calibrated operational twin without geometry, equipment, fluid, trajectory and validation data."),
        (('time series', 'forecast', 'prediction'),
         "A reliable drilling time-series workflow sorts observations chronologically, checks timestamp spacing and gaps, creates lagged/rolling features using past data only, and evaluates with chronological or walk-forward validation. Compare against persistence and moving-average baselines. Forecast horizons should be reported in samples unless the sampling interval is verified."),
        (('rop', 'rate of penetration'),
         "ROP is influenced by WOB, RPM, bit design/condition, hydraulics, formation, differential pressure, vibration, hole cleaning and operating practices. Interpret ROP with drilling context and constraints; maximizing ROP alone may increase dysfunction risk. Use the ROP Optimization module for a dataset-specific model, not as an automatic operating recommendation."),
    ]
    for keys, answer in topics:
        if any(k in q for k in keys):
            return answer
    # Dataset-aware fallback for any available column name or common parameter alias.
    aliases = {
        'hookload': ('hookload', 'hook load'), 'flow': ('flow', 'flow rate'),
        'rpm': ('rpm', 'rotary speed'), 'spp': ('spp', 'standpipe pressure'),
        'wob': ('wob', 'weight on bit'), 'depth': ('depth', 'hole depth'),
        'rop': ('rop', 'rate of penetration'), 'torque': ('torque',)
    }
    for key, words in aliases.items():
        if any(w in q for w in words) and has(frame, key):
            series = pd.to_numeric(frame[COL[key]], errors='coerce').dropna()
            if series.empty:
                return f"The dataset contains {COL[key]}, but there are no valid numeric observations to summarize."
            return (f"Dataset summary for {COL[key]}: {len(series):,} valid observations; "
                    f"mean {series.mean():.3f}, median {series.median():.3f}, "
                    f"5th–95th percentile {series.quantile(.05):.3f}–{series.quantile(.95):.3f}, "
                    f"minimum {series.min():.3f}, maximum {series.max():.3f}. "
                    "These are descriptive statistics; confirm units and operating context before interpretation.")
    return ("I can help with drilling engineering concepts and with measurements in the loaded dataset. "
            "Try asking about ECD, hole cleaning, surge/swab, torque and drag, stuck pipe, well control, "
            "MSE, Digital Twin, time-series forecasting, ROP, WOB, RPM, torque, SPP, hookload or flow. "
            "This built-in assistant uses transparent rules and local calculations; it is not a general-purpose LLM.")


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown('## 🛢️ AI DRILLING<br><span style="color:#ffad4d">INTELLIGENCE</span>',unsafe_allow_html=True)
    st.caption('ADVANCED WELL OPERATIONS ANALYTICS')
    uploaded=st.file_uploader('Upload drilling data',type=['csv','gz','xlsx','xls'],help='CSV or Excel with a timestamp and drilling measurements.')
    if 'data' not in st.session_state or uploaded is not None:
        try:
            data,source=load_data(uploaded)
            st.session_state.data=data; st.session_state.source=source
            if uploaded is not None: st.session_state.upload_name=uploaded.name
        except Exception as e:
            st.error(f'Unable to read file: {e}')
            data=st.session_state.get('data')
    else:
        data=st.session_state.data; source=st.session_state.get('source','Loaded dataset')
    page=st.radio('WORKSPACE',[
      'Executive Overview','Data Explorer','ROP Optimization','Dysfunction Intelligence',
      'Predictive Analytics','Digital Twin Simulator','Advanced Data Science','AI Drilling Assistant','Reports & Export'],
      label_visibility='collapsed')
    st.divider()
    if data is not None:
        st.caption(f'📁 {st.session_state.get("source",source)}')
        st.caption(f'{len(data):,} observations · {len(data.columns)} columns')
    st.caption('DEMONSTRATION WORKSPACE · ENGINEERING DECISION SUPPORT')

if data is None:
    st.warning('Upload a CSV or Excel drilling dataset to begin.')
    st.stop()

# ----------------------------- Executive overview -----------------------------
if page=='Executive Overview':
    hero('AI Drilling Intelligence','A unified view of drilling performance, data quality and machine-learning readiness.')
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('Observations',f'{len(data):,}')
    c2.metric('Available signals',f'{len(data.columns)-1}')
    if has(data,'date') and data[COL['date']].notna().any():
        span=data[COL['date']].max()-data[COL['date']].min(); c3.metric('Recorded duration',str(span).split('.')[0])
    else: c3.metric('Recorded duration','—')
    c4.metric('Missing cells',f'{int(data.isna().sum().sum()):,}')
    c5.metric('Duplicate rows',f'{int(data.duplicated().sum()):,}')
    st.subheader('Operational snapshot')
    cols=st.columns(4)
    for col,key,label,fmt in [(cols[0],'rop','Mean ROP','.1f'),(cols[1],'wob','Mean WOB','.1f'),(cols[2],'rpm','Mean RPM','.1f'),(cols[3],'spp','Mean SPP','.0f')]:
        if has(data,key):
            s=data[COL[key]]; col.metric(label,f'{s.mean():{fmt}}',help=f'Median: {s.median():.2f}')
        else: col.metric(label,'Not available')
    extra1,extra2,extra3=st.columns(3)
    if 'Dysfunction_Risk_Score' in data.columns:
        extra1.metric('Mean dysfunction screening score',f"{pd.to_numeric(data['Dysfunction_Risk_Score'],errors='coerce').mean():.1f}/100")
    if 'Gamma_Ray_API' in data.columns:
        extra2.metric('Mean Gamma Ray',f"{pd.to_numeric(data['Gamma_Ray_API'],errors='coerce').mean():.1f} API")
    if 'Lithology_Class' in data.columns:
        extra3.metric('Lithology classes',f"{data['Lithology_Class'].nunique(dropna=True):,}")
    left,right=st.columns([1.5,1])
    with left:
        st.markdown('#### ROP trend')
        if has(data,'date') and has(data,'rop'):
            plot_data=_chart_sample(data,8000); fig=px.line(plot_data,x=COL['date'],y=COL['rop'],color_discrete_sequence=['#27c3b5'],labels={COL['date']:'Time',COL['rop']:'ROP (m/h)'})
            st.plotly_chart(chart_layout(fig,330),width='stretch')
        elif has(data,'depth') and has(data,'rop'):
            plot_data=_chart_sample(data,8000); fig=px.scatter(plot_data,x=COL['depth'],y=COL['rop'],opacity=.45); st.plotly_chart(chart_layout(fig,330),width='stretch')
    with right:
        st.markdown('#### Data quality signals')
        q=[]
        for k in ['rop','wob','flow','rpm','torque','spp']:
            if has(data,k): q.append({'Parameter':k.upper(),'Zero %':100*(data[COL[k]].fillna(0)==0).mean(),'Missing %':100*data[COL[k]].isna().mean()})
        if q:
            qdf=pd.DataFrame(q); fig=px.bar(qdf,x='Parameter',y='Zero %',color='Zero %',color_continuous_scale='Tealgrn',text_auto='.1f'); st.plotly_chart(chart_layout(fig,330),width='stretch')
    feat_left,feat_right=st.columns(2)
    with feat_left:
        if 'Lithology_Class' in data.columns:
            st.markdown('#### Lithology distribution')
            lith=data['Lithology_Class'].fillna('Unclassified').value_counts().rename_axis('Lithology').reset_index(name='Observations')
            fig=px.bar(lith,x='Lithology',y='Observations',color='Lithology',color_discrete_sequence=['#ff9d32','#32c6df','#43d6a0','#9b8cff','#e56f91'])
            st.plotly_chart(chart_layout(fig,310),width='stretch')
    with feat_right:
        if 'Dysfunction_Risk_Score' in data.columns:
            st.markdown('#### Dysfunction screening score distribution')
            plot_data=_chart_sample(data,8000); fig=px.histogram(plot_data,x='Dysfunction_Risk_Score',nbins=35,color_discrete_sequence=['#ff9d32'])
            st.plotly_chart(chart_layout(fig,310),width='stretch')
    st.subheader('Dataset readiness and engineering notes')
    notes=[]
    if has(data,'rop'):
        z=(data[COL['rop']].fillna(0)==0).mean()*100
        notes.append(f'ROP is zero in {z:.1f}% of observations. Separate drilling from non-drilling periods before training an ROP model.')
    if has(data,'wob'):
        z=(data[COL['wob']].fillna(0)==0).mean()*100
        notes.append(f'WOB is zero in {z:.1f}% of observations. The application provides an active-drilling filter for model training.')
    if has(data,'torque'):
        med=data[COL['torque']].median(); notes.append(f'Median surface torque is {med:,.2f} in the source column unit (Klb.ft). Verify sensor scaling and engineering units before using torque in MSE or mechanical calculations.')
    notes.append('The uploaded data contains surface measurements; it does not provide confirmed dysfunction labels, lithology, bit size, or downhole vibration. Dysfunction outputs are therefore anomaly indicators, not confirmed diagnoses.')
    for n in notes: st.info(n)

# ----------------------------- Data explorer -----------------------------
elif page=='Data Explorer':
    hero('Data Explorer','Inspect measurements, select intervals and assess sensor quality before applying machine learning.')
    numeric=[c for c in data.select_dtypes(include=np.number).columns]
    with st.expander('Filter dataset',expanded=True):
        filtered=data.copy()
        if has(data,'depth'):
            lo,hi=float(data[COL['depth']].min()),float(data[COL['depth']].max())
            d1,d2=st.columns(2); a=d1.number_input('Minimum hole depth (m)',value=lo); b=d2.number_input('Maximum hole depth (m)',value=hi)
            if a<=b: filtered=filtered[filtered[COL['depth']].between(a,b)]
        if has(data,'date') and data[COL['date']].notna().any():
            t1,t2=st.columns(2); start=t1.date_input('Start date',value=data[COL['date']].min().date()); end=t2.date_input('End date',value=data[COL['date']].max().date())
            filtered=filtered[filtered[COL['date']].dt.date.between(start,end)]
    st.caption(f'{len(filtered):,} rows selected')
    if numeric:
        selected=st.multiselect('Parameters to plot',numeric,default=numeric[:min(4,len(numeric))])
        if selected:
            xcol=COL['date'] if has(filtered,'date') else (COL['depth'] if has(filtered,'depth') else None)
            if xcol:
                plot_data=_chart_sample(filtered,12000); long=plot_data[[xcol]+selected].melt(id_vars=xcol,var_name='Parameter',value_name='Value')
                fig=px.line(long,x=xcol,y='Value',color='Parameter'); st.plotly_chart(chart_layout(fig,430),width='stretch')
        st.markdown('#### Statistical profile')
        _safe_dataframe(filtered[numeric].describe().T.style.format('{:,.3f}'),width='stretch')
    st.markdown('#### Data preview')
    _safe_dataframe(filtered.head(500),width='stretch',height=320)
    st.download_button('Download filtered data (CSV)',filtered.to_csv(index=False).encode(),file_name='drilling_data_filtered.csv',mime='text/csv')

# ----------------------------- ROP optimization -----------------------------
elif page=='ROP Optimization':
    hero('AI-Based ROP Optimization','Train and compare supervised regression models, explain the drivers of ROP and explore constrained operating scenarios.')
    st.warning('Model outputs are statistical estimates from the uploaded dataset. They are not guaranteed operating setpoints. Apply engineering limits, bit/BHA constraints and well-control requirements independently.')
    c1,c2=st.columns([1,2])
    with c1:
        model_name=st.selectbox('Regression algorithm',['Random Forest','Extra Trees','Histogram Gradient Boosting'])
        active_only=st.checkbox('Train on active drilling observations (ROP, WOB and RPM > 0)',value=True)
        if st.button('Train / refresh ROP model',type='primary'):
            with st.spinner('Training model using chronological train/test split...'):
                try:
                    result=train_rop_model(data,model_name,active_only)
                    model,features,metrics,preds,Xtr,Xte,ytr,yte=result
                    st.session_state.rop_model={'model':model,'features':features,'metrics':metrics,'preds':preds,'Xtr':Xtr,'Xte':Xte,'yte':yte,'name':model_name}
                    st.success('Model trained and evaluated.')
                except Exception as e: st.error(str(e))
    with c2:
        st.markdown('#### Model evaluation')
        saved=st.session_state.get('rop_model')
        if saved:
            m=saved['metrics']; a,b,c=st.columns(3); a.metric('R²',f"{m['R²']:.3f}"); b.metric('MAE',f"{m['MAE']:.2f} m/h"); c.metric('RMSE',f"{m['RMSE']:.2f} m/h")
            st.caption(f"{saved['name']} · chronological 80/20 split · {m['Train rows']:,} train / {m['Test rows']:,} test rows")
            p=saved['preds'].reset_index(drop=True).reset_index(names='Test observation')
            fig=go.Figure(); fig.add_trace(go.Scatter(x=p['Test observation'],y=p['Actual ROP'],mode='lines',name='Actual')); fig.add_trace(go.Scatter(x=p['Test observation'],y=p['Predicted ROP'],mode='lines',name='Predicted'))
            st.plotly_chart(chart_layout(fig,330),width='stretch')
            imp=get_model_importance(saved['model'],saved['features'],saved['Xte'],saved['yte'])
            if not imp.empty:
                fig=px.bar(imp.sort_values('Importance'),x='Importance',y='Feature',orientation='h',color='Importance',color_continuous_scale='Teal'); st.plotly_chart(chart_layout(fig,280),width='stretch')
        else: st.info('Train a model to see evaluation metrics, predictions and feature importance.')
    if st.session_state.get('rop_model'):
        saved=st.session_state.rop_model
        st.divider(); st.subheader('Constrained operating scenario explorer')
        st.caption('The scenario explorer searches within the observed training-data envelope. It does not establish safe limits; enter engineering-approved limits before using results.')
        model=saved['model']; Xtr=saved['Xtr']; feature=list(Xtr.columns)
        scenario={}
        cols=st.columns(min(3,len(feature)))
        for i,f in enumerate(feature):
            mn=float(Xtr[f].quantile(.05)); mx=float(Xtr[f].quantile(.95)); md=float(Xtr[f].median())
            if not np.isfinite(mn) or not np.isfinite(mx) or not np.isfinite(md):
                mn, mx, md = 0.0, 1.0, 0.5
            with cols[i%len(cols)]:
                scenario[f]=st.slider(f, min_value=float(mn),max_value=float(mx) if mx>mn else float(mn+1),value=float(np.clip(md,mn,mx if mx>mn else mn+1)))
        if st.button('Estimate scenario ROP'):
            pred=float(model.predict(pd.DataFrame([scenario]))[0]); st.metric('Estimated ROP',f'{pred:.2f} m/h')
            st.json({k:round(v,3) for k,v in scenario.items()})
        st.markdown('#### WOB–RPM response surface')
        if COL['wob'] in feature and COL['rpm'] in feature:
            other={f:float(Xtr[f].median()) for f in feature}
            wlo,whi=Xtr[COL['wob']].quantile([.05,.95]); rlo,rhi=Xtr[COL['rpm']].quantile([.05,.95])
            wvals=np.linspace(wlo,whi,22); rvals=np.linspace(rlo,rhi,22); grid=[]
            for w in wvals:
                for r in rvals:
                    row=other.copy(); row[COL['wob']]=w; row[COL['rpm']]=r; grid.append(row)
            z=model.predict(pd.DataFrame(grid)); zz=z.reshape(len(wvals),len(rvals))
            fig=go.Figure(data=go.Contour(x=rvals,y=wvals,z=zz,colorscale='Viridis',colorbar_title='ROP m/h'))
            fig.update_layout(xaxis_title='Rotary speed (rpm)',yaxis_title='WOB (Klb)'); st.plotly_chart(chart_layout(fig,430),width='stretch')

# ----------------------------- Dysfunction intelligence -----------------------------
elif page=='Dysfunction Intelligence':
    hero('Dysfunction Intelligence','Detect unusual surface drilling behavior using unsupervised learning and transparent engineering indicators.')
    st.warning('Anomaly detection identifies unusual patterns, not the physical cause. Confirm alerts against rig reports, downhole tools, drilling practices and offset-well context.')
    if 'Dysfunction_Risk_Score' in data.columns:
        st.markdown('#### Pre-computed screening indicators')
        score=pd.to_numeric(data['Dysfunction_Risk_Score'],errors='coerce')
        a,b,c=st.columns(3)
        a.metric('Mean screening score',f'{score.mean():.1f}/100' if score.notna().any() else '—')
        c.metric('Rows with indicators',f'{int((score>0).sum()):,}' if score.notna().any() else '—')
        if 'Dysfunction_Screening_Status' in data.columns:
            b.metric('Screening categories',f'{data["Dysfunction_Screening_Status"].nunique(dropna=True):,}')
        if 'Dysfunction_Screening_Status' in data.columns:
            counts=data['Dysfunction_Screening_Status'].fillna('Unclassified').value_counts().rename_axis('Screening status').reset_index(name='Observations')
            fig=px.bar(counts,x='Screening status',y='Observations',color='Screening status',color_discrete_sequence=['#ff9d32','#32c6df','#43d6a0','#9b8cff'])
            st.plotly_chart(chart_layout(fig,300),width='stretch')
        st.caption('These are pre-computed demonstration screening indicators. They are not validated downhole event labels or confirmed diagnoses.')
    keys=[k for k in ['rop','wob','rpm','torque','hookload','spp','flow','depth'] if has(data,k)]
    default=[k for k in ['rop','wob','rpm','torque','hookload','spp','flow'] if k in keys]
    chosen=st.multiselect('Signals for anomaly detection',[COL[k] for k in keys],default=[COL[k] for k in default])
    contamination=st.slider('Expected anomaly fraction (algorithm setting)',.005,.15,.03,.005)
    if st.button('Run Isolation Forest',type='primary'):
        x=data[chosen].replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)>100 and len(chosen)>=2:
            scaler=StandardScaler(); xs=scaler.fit_transform(x)
            detector=IsolationForest(n_estimators=250,contamination=contamination,random_state=42,n_jobs=-1)
            labels=detector.fit_predict(xs); scores=-detector.score_samples(xs)
            out=data.loc[x.index].copy(); out['Anomaly']=(labels==-1); out['Anomaly score']=scores
            st.session_state.anomaly=out
            st.success(f'Analysis completed on {len(out):,} valid observations.')
        else: st.error('Select at least two signals and provide at least 100 complete observations.')
    if 'anomaly' in st.session_state:
        out=st.session_state.anomaly; n=int(out['Anomaly'].sum()); a,b,c=st.columns(3); a.metric('Flagged observations',f'{n:,}'); b.metric('Flagged share',f'{100*n/len(out):.2f}%'); c.metric('Analyzed rows',f'{len(out):,}')
        xcol=COL['date'] if has(out,'date') else (COL['depth'] if has(out,'depth') else None)
        if xcol and has(out,'rop'):
            fig=px.scatter(out,x=xcol,y=COL['rop'],color='Anomaly',color_discrete_map={True:'#ff6b6b',False:'#27c3b5'},opacity=.65,hover_data=['Anomaly score'])
            st.plotly_chart(chart_layout(fig,380),width='stretch')
        st.markdown('#### Highest anomaly scores')
        _safe_dataframe(out.nlargest(100,'Anomaly score'),width='stretch')
        st.download_button('Download anomaly results',out.to_csv(index=False).encode(),file_name='drilling_anomaly_results.csv',mime='text/csv')
    st.divider(); st.subheader('Engineering trend indicators')
    if has(data,'date'):
        d=data.copy().set_index(COL['date'])
        for k in ['torque','rop','spp','hookload']:
            if has(data,k): d[f'{COL[k]} rolling std'] = d[COL[k]].rolling('5min',min_periods=5).std()
        pick=[c for c in d.columns if 'rolling std' in c]
        if pick:
            sel=st.selectbox('Select variability indicator',pick); fig=px.line(d.reset_index(),x=COL['date'],y=sel); st.plotly_chart(chart_layout(fig,300),width='stretch')

# ----------------------------- Predictive analytics -----------------------------
elif page=='Predictive Analytics':
    hero('Predictive Analytics','Build chronological time-series models for short-horizon drilling parameter forecasting.')
    st.info('This first implementation uses lagged surface measurements and gradient boosting for one-step-ahead prediction. It is a baseline forecasting workflow, not a validated real-time control system.')
    target_key=st.selectbox('Target variable',['rop','torque','spp','hookload','wob','rpm'],format_func=lambda k:COL[k],index=0)
    available=[k for k in ['rop','torque','spp','hookload','wob','rpm','flow','depth'] if has(data,k)]
    lag_keys=st.multiselect('Input signals',available,default=[k for k in [target_key,'wob','rpm','torque','spp','flow'] if k in available])
    if st.button('Train time-series baseline',type='primary'):
        if target_key not in lag_keys: lag_keys=[target_key]+lag_keys
        base=data.copy().sort_values(COL['date']) if has(data,'date') else data.copy()
        target=COL[target_key]; feats=[COL[k] for k in lag_keys]
        base=base[feats].replace([np.inf,-np.inf],np.nan)
        X=add_time_features(base,target,feats)
        ds=pd.concat([base[target].rename('target'),X],axis=1).dropna()
        split=int(len(ds)*.8); Xtr,Xte=ds.drop(columns='target').iloc[:split],ds.drop(columns='target').iloc[split:]; ytr,yte=ds.target.iloc[:split],ds.target.iloc[split:]
        if len(ds)<200 or len(Xte)<20: st.error('Insufficient valid time-series observations after lag generation.')
        else:
            mdl=HistGradientBoostingRegressor(max_iter=160,learning_rate=.06,max_leaf_nodes=18,l2_regularization=1.0,random_state=42); mdl.fit(Xtr,ytr); pr=mdl.predict(Xte)
            st.session_state.ts_result={'model':mdl,'target':target,'features':list(Xtr.columns),'actual':yte,'pred':pr,'mae':mean_absolute_error(yte,pr),'rmse':mean_squared_error(yte,pr)**.5,'r2':r2_score(yte,pr)}
            st.success('Chronological time-series model trained.')
    if 'ts_result' in st.session_state:
        r=st.session_state.ts_result; a,b,c=st.columns(3); a.metric('MAE',f"{r['mae']:.3f}"); b.metric('RMSE',f"{r['rmse']:.3f}"); c.metric('R²',f"{r['r2']:.3f}")
        fig=go.Figure(); fig.add_trace(go.Scatter(y=r['actual'].values,name='Actual',mode='lines')); fig.add_trace(go.Scatter(y=r['pred'],name='Predicted',mode='lines')); st.plotly_chart(chart_layout(fig,370),width='stretch')
        st.caption(f"Target: {r['target']} · one-step-ahead prediction using prior samples. Test partition is chronological.")

# ----------------------------- Digital twin -----------------------------
elif page=='Digital Twin Simulator':
    hero('Digital Twin Simulator','Explore model-based what-if scenarios using the learned relationship between surface parameters and ROP.')
    st.warning('This is a data-driven surrogate model, not a physics-calibrated well/rig digital twin. Scenario outputs are only meaningful within the observed data envelope and do not replace operational procedures.')
    saved=st.session_state.get('rop_model')
    if not saved:
        st.info('Train an ROP model in the ROP Optimization module first.'); st.stop()
    model=saved['model']; Xtr=saved['Xtr']; features=list(Xtr.columns)
    st.markdown('#### Configure a simulated operating point')
    scenario={}; input_errors=[]; cols=st.columns(3)
    for i,f in enumerate(features):
        lo=float(Xtr[f].quantile(.05)); hi=float(Xtr[f].quantile(.95)); med=float(Xtr[f].median())
        if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(med):
            lo, hi, med = 0.0, 1.0, 0.5
        upper=hi if hi>lo else lo+1
        default=float(np.clip(med,lo,upper))
        # Text inputs keep the displayed digits in Western/English form regardless of browser locale.
        with cols[i%3]:
            raw_value=st.text_input(f, value=f"{default:.2f}", key=f"digital_twin_input_{i}")
        try:
            parsed=float(raw_value.strip().replace(",", ""))
            if not np.isfinite(parsed):
                raise ValueError("Enter a finite number.")
            if parsed < lo or parsed > upper:
                input_errors.append(f"{f}: enter a value between {lo:.2f} and {upper:.2f}.")
            scenario[f]=parsed
        except (TypeError, ValueError):
            input_errors.append(f"{f}: enter a valid number using English digits and a decimal point.")
    if input_errors:
        for error in input_errors: st.error(error)
    if st.button('Run digital twin scenario',type='primary',disabled=bool(input_errors)):
        prediction=float(model.predict(pd.DataFrame([scenario]))[0]); st.session_state.twin={'input':scenario,'rop':prediction}
    if 'twin' in st.session_state:
        twin=st.session_state.twin; st.metric('Model-estimated ROP',f"{twin['rop']:.2f} m/h")
        st.json(twin['input'])
        st.caption('Compare this estimate with actual operational constraints, offset data and engineering judgement before considering any change.')

# ----------------------------- Advanced data science -----------------------------
elif page=='Advanced Data Science':
    hero('Advanced Data Science Laboratory','Correlation, dimensionality reduction and unsupervised operating-pattern discovery.')
    keys=sorted(data.select_dtypes(include=np.number).columns.tolist())
    chosen=st.multiselect('Select numerical signals (all available numeric fields)', keys,default=[COL[k] for k in ['rop','wob','rpm','torque','hookload','spp','flow'] if has(data,k)])
    if len(chosen)>=2:
        clean=data[chosen].replace([np.inf,-np.inf],np.nan).dropna()
        st.markdown('#### Correlation matrix')
        corr=clean.corr(method='pearson'); fig=px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,aspect='auto'); st.plotly_chart(chart_layout(fig,520),width='stretch')
        st.markdown('#### PCA – dominant operating patterns')
        scaled=StandardScaler().fit_transform(clean); pcs=PCA(n_components=2,random_state=42).fit_transform(scaled)
        pcdf=pd.DataFrame({'PC1':pcs[:,0],'PC2':pcs[:,1]})
        fig=px.scatter(pcdf.sample(min(6000,len(pcdf)),random_state=42),x='PC1',y='PC2',opacity=.5); st.plotly_chart(chart_layout(fig,390),width='stretch')
        st.markdown('#### K-Means operating-pattern discovery')
        k=st.slider('Number of clusters',2,8,4)
        if st.button('Run K-Means clustering'):
            km=KMeans(n_clusters=k,n_init=10,random_state=42); labels=km.fit_predict(scaled)
            result=clean.copy(); result['Cluster']=labels
            st.session_state.clusters=result
            st.success('Clustering completed. Cluster labels are statistical groups, not confirmed drilling activities.')
        if 'clusters' in st.session_state:
            cl=st.session_state.clusters; _safe_dataframe(cl.groupby('Cluster').mean(numeric_only=True).round(3),width='stretch')
            st.download_button('Download cluster assignments',cl.to_csv(index=False).encode(),file_name='drilling_clusters.csv',mime='text/csv')

# ----------------------------- AI assistant -----------------------------
elif page=='AI Drilling Assistant':
    hero('AI Drilling Assistant','A free, local engineering assistant for drilling concepts and dataset-aware questions.')
    st.info('No paid API or external AI service is used. Answers are generated by a transparent engineering knowledge layer and local dataset calculations; they are not a live LLM.')
    with st.form('engineering_assistant_form'):
        question=st.text_area('Ask a drilling engineering or data question',placeholder='Example: Explain ECD and what inputs are needed to calculate it.',height=100)
        submitted=st.form_submit_button('Submit question',type='primary')
    if submitted and question.strip():
        answer=engineering_knowledge_answer(question.strip(),data)
        st.session_state.setdefault('assistant_history',[]).append({'Question':question.strip(),'Answer':answer})
    if st.session_state.get('assistant_history'):
        for item in reversed(st.session_state.assistant_history[-8:]):
            with st.expander(item['Question'],expanded=True):
                st.write(item['Answer'])
        if st.button('Clear conversation'):
            st.session_state.assistant_history=[]
            st.rerun()
    st.caption('Engineering guidance is general information. Apply approved well-specific procedures, verify units and assumptions, and obtain operational authorization before field decisions.')

# ----------------------------- Reports -----------------------------
elif page=='Reports & Export':
    hero('Reports & Export','Package dataset quality, performance statistics and available model outputs for engineering review.')
    st.markdown('#### Generate data summary')
    if st.button('Prepare report tables',type='primary'):
        summary=data.describe(include='all').T
        quality=pd.DataFrame({'Missing':data.isna().sum(),'Missing %':100*data.isna().mean(),'Unique values':data.nunique(),'Zero count':[int((data[c]==0).sum()) if pd.api.types.is_numeric_dtype(data[c]) else np.nan for c in data.columns]})
        st.session_state.report_summary=summary; st.session_state.report_quality=quality
    if 'report_summary' in st.session_state:
        st.markdown('**Statistical summary**'); _safe_dataframe(st.session_state.report_summary,width='stretch')
        st.markdown('**Data quality report**'); _safe_dataframe(st.session_state.report_quality,width='stretch')
        out=io.BytesIO()
        with pd.ExcelWriter(out,engine='openpyxl') as writer:
            data.to_excel(writer,index=False,sheet_name='Source Data')
            st.session_state.report_summary.to_excel(writer,sheet_name='Statistics')
            st.session_state.report_quality.to_excel(writer,sheet_name='Data Quality')
            if 'anomaly' in st.session_state: st.session_state.anomaly.to_excel(writer,index=False,sheet_name='Anomaly Results')
            if 'clusters' in st.session_state: st.session_state.clusters.to_excel(writer,index=False,sheet_name='Clusters')
        st.download_button('Download Excel engineering pack',out.getvalue(),file_name='AI_Drilling_Intelligence_Report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.download_button('Download source data CSV',data.to_csv(index=False).encode(),file_name='drilling_source_data.csv',mime='text/csv')
    st.caption('PDF export can be added as a report template extension. Excel export is enabled in this version.')

st.markdown('---')
st.markdown('<div class="small-note">AI Drilling Intelligence · Demonstration dataset generated for analytics and software testing; not field measurements · Validate units, sensor quality, model performance and operational limits before field application.</div>',unsafe_allow_html=True)
