#!/usr/bin/env python
# coding: utf-8

# In[46]:

# In[ ]:


import subprocess
import requests
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.dates as mdates
import scipy.optimize as opt
from sklearn.utils import resample
from scipy.spatial.distance import pdist,squareform
from sklearn import preprocessing
from sklearn.utils import resample

# Streamlit setup - nom de la page web
st.set_page_config(page_title="European stocks selector & optimizer", layout="wide")

# In[85]:


def extract_index_components(wiki_url, index_name):
    headers = {
        "User-Agent": "MyPythonScript/1.0 (contact@example.com)"
    }
    response = requests.get(wiki_url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(response.text)

    for table in tables:
        if 'Company' in table.columns or 'Security' in table.columns or 'Name' in table.columns:
            companies = []
            for _, row in table.iterrows():
                company_info = {
                    "company": row.get("Company") or row.get("Security") or row.get("Name"),
                    "ticker": row.get("Ticker") or row.get("Symbol") or row.get("Ticker symbol") or None
                }
                companies.append(company_info)
            return companies
    return None



@st.cache_data(ttl=86400)
def extract_corpos(indices):
    for index_name, url in indices.items():
            companies = extract_index_components(url, index_name)
    
            if index_name == "CAC 40":
                cac40_name = [c["company"] for c in companies]
                cac40_ticker = [c["ticker"] for c in companies]
            elif index_name == "DAX 40":
                dax40_name = [c["company"] for c in companies]
                dax40_ticker = [c["ticker"] for c in companies]
            elif index_name == "AEX 25":
                aex25_name = [c["company"] for c in companies]
                aex25_ticker = [c["ticker"] for c in companies]
            elif index_name == "IBEX 35":
                ibex35_name = [c["company"] for c in companies]
                ibex35_ticker = [c["ticker"] for c in companies]
            elif index_name == "FTSE MIB":
                ftsemib_name = [c["company"] for c in companies]
                ftsemib_ticker = [c["ticker"] for c in companies]
    
    for i in range(len(aex25_ticker)):
        l=aex25_ticker[i]+".AS"
        aex25_ticker[i]=l
    
    t=cac40_ticker+dax40_ticker+aex25_ticker+ibex35_ticker+ftsemib_ticker
    tickers=set(t)
    tickers=list(tickers)
    
    c=cac40_name+dax40_name+aex25_name+ibex35_name+ftsemib_name
    corpo=set(c)
    corpo=list(corpo)
    return tickers, t,c
    



# In[112]:
# OPTIMIZATION ROUTINES
#Volatility minimization with return objective
def port_minvol_ro(mean, cov, ro):
    def objective(W, R, C, ro):
        # calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov),W)
        #objective: min vol
        util=varp**0.5
        return util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. } , {'type':'eq', 'fun': lambda W: np.dot(W.T,mean)-ro })
    optimized=opt.minimize(objective,W,(mean,cov,ro),
                                      method='SLSQP',constraints=c_,bounds=b_, options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

#Volatility minimization
def port_minvol(mean, cov):
    def objective(W, R, C):
        # calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov),W)
        #objective: min vol
        util=varp**0.5
        return util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n                 
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    optimized=opt.minimize(objective,W,(mean,cov),
                                      method='SLSQP',constraints=c_,bounds=b_,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

#Return maximization
def port_maxret(mean, cov):
    def objective(W, R, C):
        # calculate mean/variance of the portfolio
        meanp=np.dot(W.T,mean)
        #objective: Max return
        util=1/meanp
        return util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n                 
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    optimized=opt.minimize(objective,W,(mean,cov),
                                      method='SLSQP',constraints=c_,bounds=b_,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

# In[105]:

@st.cache_data(ttl=86400)
def old_tickers(yea,tick):
    today = date.today()
    ten_years_ago = today - relativedelta(years=yea)
    
    valid_tickers = []
    for t in tick:
        df = yf.download(t, period="max", progress=False,auto_adjust=True)
        first_date = df.index.min().date()
        if first_date <= ten_years_ago:
            valid_tickers.append(t)

    return valid_tickers

@st.cache_data(ttl=86400)
def EUR_tickers(tick):
    good=[]
    for t in tick:
        currency = yf.Ticker(t).info.get("currency", "Unknown")
        if currency=="EUR":
            good.append(t)
     
    return good


# In[114]:

@st.cache_data(ttl=86400)
def extract_data(valid_tickers):
    data = pd.DataFrame()
    today = date.today()
    thirty_years_ago = date.today() - relativedelta(years=30)
    
    for ti in valid_tickers:
        #df = yf.Ticker(t).history(period="max")["Close"]
        df=yf.download(ti, start=thirty_years_ago, end=today, progress=False,auto_adjust=True)["Close"]
        df.name = ti
        data = pd.concat([data, df], axis=1)

    
    return data

# In[36]
def efficient_frontier(mean,sigma, truemean,truesigma):
    weights_matrix = np.zeros((20, nb_assets+2))
    v=np.zeros((1,40))
    
    min_weights_RS=port_minvol(mean, sigma)
    min_ptfmean_RS=min_weights_RS.T@mean
    min_ptfsig_RS=(min_weights_RS.T@sigma@min_weights_RS)**0.5
    
    max_weights_RS=port_maxret(mean, sigma)
    max_ptfmean_RS=max_weights_RS.T@mean
    max_ptfsig_RS=(max_weights_RS.T@sigma@max_weights_RS)**0.5
    
    step_RS=(max_ptfmean_RS-min_ptfmean_RS)/19
    
    for i in range(20):
        ro=min_ptfmean_RS+i*step_RS
        wei_ro_RS=port_minvol_ro(mean, sigma, ro)
        mean_ro_RS=wei_ro_RS.T@truemean
        sig_ro_RS=(wei_ro_RS.T@truesigma@wei_ro_RS)**0.5
        weights_matrix[i,:-2]=wei_ro_RS
        weights_matrix[i,-2]=mean_ro_RS
        weights_matrix[i,-1]=sig_ro_RS
        v[0,2*i]=mean_ro_RS
        v[0,2*i+1]=sig_ro_RS
    return weights_matrix,v

def bootstrap(returns):
    return resample(returns, replace=True, n_samples=None)


@st.cache_data(ttl=86400)
def simul_EF(returns,mean,cov):
    nb_simul=100
    weights_matrix_S = np.zeros((20, nb_assets+2,nb_simul))
    vect=np.zeros((nb_simul,40))
    i=0

    while i<nb_simul:
        sm=bootstrap(returns)
        meanS=np.average(sm, axis=0)
        sigmaS=np.cov(sm,rowvar=False)
        weights_matrix_S[:,:,i],vect[i,:]=efficient_frontier(meanS,sigmaS,meanS,sigmaS)
        i=i+1
    
    standard=preprocessing.scale(vect,axis=1)

    # pairwise Euclidean distances
    D = squareform(pdist(standard, metric='euclidean'))
    # sum distances for each point
    sum_dist = D.sum(axis=1)
    # clusteroid index = the most central bootstrap
    clusteroid_idx = np.argmin(sum_dist)
    # the clusteroid feature vector
    clusteroid = vect[clusteroid_idx]

    RES_weights=weights_matrix_S[:,:,clusteroid_idx]
    ptf = {}
    for i in range(20):
        ptf[i] = RES_weights[i, :-2]
    
    INV = np.linalg.pinv(np.cov(vect,rowvar=False))
    distance=np.zeros((nb_simul,2))

    for i in range(nb_simul):
        v=vect[i,:]
        distance[i,0]=int(i)
        distance[i,1]=((v-clusteroid).T@INV@(v-clusteroid))**0.5
    
    col=-1
    threshold = np.percentile(distance[:, col], 95)
    sim_indices = distance[:,0][distance[:, col] <= threshold]
    
    subset_efficient = weights_matrix_S[:, :, np.array(sim_indices, dtype=int)]

    #test & plot
    original_EF,ef=efficient_frontier(mean,cov,mean,cov)
    original_EF=original_EF[:-2]
    col_x, col_y = -1, -2  # the two columns you want from the 2nd dimension

    n_slices = subset_efficient.shape[2]

    plt.figure(figsize=(6, 5))

    for i in range(n_slices):
        plt.scatter(subset_efficient[:, col_x, i], subset_efficient[:, col_y, i], color="blue")

    plt.plot(RES_weights[:, col_x], RES_weights[:, col_y], color="red")
    plt.plot(original_EF[:, col_x], original_EF[:, col_y], color="green")

    plt.xlabel("vol")
    plt.ylabel("mean")
    plt.title("95% simulated efficient frontiers + clusteroid")
    plt.grid(True)

   #st.pyplot(plt)
    return ptf[0],ptf[4],ptf[9],ptf[14],ptf[19]


def weights_plots_minv(type,weights):
    today = date.today()


    ##### ptf returns

    portfolio_returns = used_returns.dot(weights)
    portfolio_px=used_px.dot(weights)

    portfolio_annualized_return=np.mean(portfolio_returns)*252
    portfolio_annualized_vol=np.std(portfolio_returns)*np.sqrt(252)

    st.title(type+" Portfolio")
    st.write("")
    st.write("Portfolio's annualized return : "+str(round(portfolio_annualized_return*100,2))+"%")
    st.write("Portfolio's annualized risk : "+str(round(portfolio_annualized_vol*100,2))+"%")



    # -------------------------------
    # Affichage des poids
    # -------------------------------

    df_table_R = pd.DataFrame({"Repartition": [f"{w:.1%}" for w in weights]}, index=assets_names)
    df_table_R.index.name = "Stock"
    #st.write(assets_names)
    # Format DataFrame
    html = df_table_R.T.to_html(
    index=True,
    justify='center',
    border=0,
    classes='styled-table')

    # CSS Styling
    st.markdown(f"""
    <style>
    .nice-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 14px;
        font-family: Arial, sans-serif;
        text-align: center;
    }}
    .nice-table th, .nice-table td {{
        border: 1px solid #d0d0d0;
        padding: 8px 10px;
        width: {100 / len(df_table_R.columns)}%;
    }}
    .nice-table th {{
        background-color: #f5f5f5;
        color: #222;
        font-weight: 600;
    }}
    .nice-table tr:nth-child(even) {{
        background-color: #fafafa;
    }}
    .nice-table tr:hover {{
        background-color: #e8f0fe;
        transition: background-color 0.2s ease;
    }}
    /* Dark mode adjustments */
    [data-testid="stAppViewContainer"][class*="dark"] .nice-table th {{
        background-color: #333 !important;
        color: #fff !important;
    }}
    [data-testid="stAppViewContainer"][class*="dark"] .nice-table td {{
        border-color: #555 !important;
        color: #ddd !important;
    }}
    [data-testid="stAppViewContainer"][class*="dark"] .nice-table tr:nth-child(even) {{
        background-color: #2a2a2a !important;
    }}
    [data-testid="stAppViewContainer"][class*="dark"] .nice-table tr:hover {{
        background-color: #3a3a3a !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Portfolio Repartition")
    st.markdown(html, unsafe_allow_html=True)
    
    ################### Choix de l’horizon
    
    ## Initialize session state
    horizon="MAX"
    periods = ["1M", "6M", "1Y", "5Y", "10Y", "20Y", "MAX"]

    # Selection of the horizon
    st.write("")
    horizon = st.radio("Choose horizon:", periods, horizontal=True)


    ###################### Filtration de la data historique selon l’horizon
    if horizon != "MAX":
        if horizon[-1]=="Y":
            target = date.today() - relativedelta(years=int(horizon[:-1]))
            closest_idx=portfolio_px.index.get_indexer([target],method='nearest')[0]
            closest_date=portfolio_px.index[closest_idx]
            cum_perf = portfolio_px[portfolio_px.index >= closest_date]
        elif horizon[-1]=="M":
            target = date.today() - relativedelta(months=int(horizon[:-1]))
            closest_idx=portfolio_px.index.get_indexer([target],method='nearest')[0]
            closest_date=portfolio_px.index[closest_idx]
            cum_perf = portfolio_px[portfolio_px.index >= closest_date]
    else:
        cum_perf=portfolio_px

    # -------------------------------
    # Affichage du graphique
    # -------------------------------
    # Create two columns
    
    col1, col2 = st.columns(2)
    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)

    # --- Column 1: historical price Chart ---
    with col1:

        # vérification de si la performance sur l'hozrizon selectionné est positive ou négative
        curr=cum_perf[-1]
        fst=cum_perf[1]
        delta=curr-fst
        if delta>0:
            sub="+ "+str(round(delta,2))+" € in "+horizon+" : +"+str(round((delta/fst)*100,2))+" %"
            color="green"
            cl="Greens"
        else:
            sub=str(round(delta,2))+" € in "+horizon+" : "+str(round((delta/fst)*100,2))+" %"
            color="red"
            cl="Reds"
        
        #plot

        st.subheader("Historical Portfolio's price evolution")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.lineplot(x=cum_perf.index, y=cum_perf.values, ax=ax1, color=color, linewidth=2)
    
        #Titles
        fig1.suptitle("Portfolio : "+str(round(curr,2))+" € ", fontsize=15, fontweight="bold",ha="center")
        ax1.set_title(sub, fontsize=12, color=color, fontstyle='italic',pad=15,ha="center")

        #mise en forme du graph
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xlabel("Date", fontsize=10, fontstyle='italic', labelpad=10)
        ax1.set_ylabel("Price", fontsize=10, fontstyle='italic', labelpad=10)
        if ((horizon=="20Y")|(horizon=="10Y")|(horizon=="MAX")):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif (horizon=="1M"):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        
        locator = mdates.AutoDateLocator(maxticks=6)  # Show at most x labels
        ax1.xaxis.set_major_locator(locator)
        #plt.xticks(rotation=45)
        ax1.grid(False)
        ax1.legend(loc='lower right', frameon=False)
        
        st.pyplot(fig1)
        
    # --- Column 2: Pie Chart ---
    with col2:
        st.subheader("Portfolio Repartition")
        fig2, ax2 = plt.subplots(figsize=(8,10))

        # Filter weights > threshold
        threshold = 0.002  # 0.2%
        filtered_weights = []
        filtered_assets = []
        for w, a in zip(weights, assets_names):
            if w > threshold:
                filtered_weights.append(w)
                filtered_assets.append(a)

        # Pie chart with pastel colors and percentage labels
        wedges, texts, autotexts = ax2.pie(
            filtered_weights, 
            labels=None, 
            autopct="%1.1f%%",
            startangle=90, 
            colors=sns.color_palette(cl, len(filtered_weights)),
            textprops={'fontsize': 9}
        )
        
        ax2.legend(wedges,filtered_assets,
        title="Assets",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Draw circle for donut style (optional)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig2.gca().add_artist(centre_circle)
        st.pyplot(fig2)

# In[37]:


#### main

indices = {
    "CAC 40": "https://en.wikipedia.org/wiki/CAC_40",
    "DAX 40": "https://en.wikipedia.org/wiki/DAX",
    "AEX 25":"https://en.wikipedia.org/wiki/AEX_index",
    "IBEX 35":"https://en.wikipedia.org/wiki/IBEX_35",
    "FTSE MIB":"https://en.wikipedia.org/wiki/FTSE_MIB"
    
}

tickers,tick,corpo=extract_corpos(indices)
valid_tickers=EUR_tickers(old_tickers(20,tickers))

for index_name, url in indices.items():
    companies = extract_index_components(url, index_name)
    
    if index_name == "CAC 40":
        cac40_name = [c["company"] for c in companies]
        cac40_ticker = [c["ticker"] for c in companies]
    elif index_name == "DAX 40":
        dax40_name = [c["company"] for c in companies]
        dax40_ticker = [c["ticker"] for c in companies]
    elif index_name == "AEX 25":
        aex25_name = [c["company"] for c in companies]
        aex25_ticker = [c["ticker"] for c in companies]
    elif index_name == "IBEX 35":
        ibex35_name = [c["company"] for c in companies]
        ibex35_ticker = [c["ticker"] for c in companies]
    elif index_name == "FTSE MIB":
        ftsemib_name = [c["company"] for c in companies]
        ftsemib_ticker = [c["ticker"] for c in companies]

dico_index={}
for i in range(len(tick)):
    dico_index[tick[i]]=corpo[i]

data=extract_data(valid_tickers)
data=data.rename(columns=dico_index)

data=data.loc[:,~data.columns.duplicated()]

data_c=data.dropna(axis=0)
data_c.sort_index(inplace=True)
returns=data_c.pct_change().dropna()

st.title("Portfolio Optimization")

selected_display = st.empty()
st.write("")
# Dictionnaire indices -> tickers
index_to_actions = {}

index_to_actions = {
    "CAC 40": cac40_name,
    "DAX 40": dax40_name,
    "AEX 25": aex25_name,
    "IBEX 35": ibex35_name,
    "FTSE MIB": ftsemib_name
}

selected_companies = []
for index_name, actions in index_to_actions.items():
    with st.expander(index_name):
        selected = st.multiselect(f"Select stocks from {index_name}", options=actions, key=f"ms_{index_name}")
        selected_companies.extend(selected)

if selected_companies:
    selected_display.markdown(
        "<h4>Selected Stocks:</h4>" +
        "".join([
            f"<span style='font-size:15px; background-color:#f0f2f6; "
            f"border-radius:10px; padding:5px 10px; margin:3px; "
            f"display:inline-block;'>{item}</span>"
            for item in selected_companies
        ]),
        unsafe_allow_html=True
    )
else:
    selected_display.markdown("No stocks selected yet.")
selected=selected_companies

# --- Vérification du nombre sélectionné ---

# Initialize validation flag
if "validated" not in st.session_state:
    st.session_state.validated = False
# Initialize validation flag
if "submitt" not in st.session_state:
    st.session_state.submitt = False

# --- Validation logic ---

validate = st.button("✅ Validate Selection")

if validate:
    if len(selected) == 0:
        st.info("Please select at least 1 stock before validation")
        st.session_state.validated = False
    elif len(selected) > 10:
        st.error("❌ You can select up to 10 stocks")
        st.session_state.validated = False
    else:
        st.success(f"✅ You selected {len(selected)} stock(s).")
        st.session_state.validated = True  # Persist validation across reruns

# --- After validation, run your plots function ---
if st.session_state.validated:

    ##isolation of selected assets in the DF
    used_returns = returns.loc[:, returns.columns.isin(selected)]
    used_px = data_c.loc[:, data_c.columns.isin(selected)]

    ###optimization
    
    selected_returns=np.array(used_returns)
    assets_names=used_returns.columns
    nb_assets=len(assets_names)
    nb_obs=selected_returns.shape[0]

    mean=np.average(selected_returns, axis=0)
    covariance_matrix=np.cov(selected_returns,rowvar=False)
    #st.write(selected_returns.shape)
    #used_returns.to_excel('portfolio_results.xlsx', index=False)


    ####détermination des poids
    if len(selected)==1:
        w_1stock = np.zeros(len(selected))
        w_1stock[0] = 1
        weights_plots_minv("Single Stock",w_1stock)

    else:

        weights_MV,weights_S1,weights_S2,weights_S3,weights_MaxRet=simul_EF(selected_returns,mean,covariance_matrix)
        
        portfolio_returns_MV = used_returns.dot(weights_MV)
        portfolio_returns_MR=used_returns.dot(weights_MaxRet)
        portfolio_annualized_return_MV=np.mean(portfolio_returns_MV)*252
        portfolio_annualized_vol_MV=np.std(portfolio_returns_MV)*np.sqrt(252)
        portfolio_annualized_return_MR=np.mean(portfolio_returns_MR)*252
        portfolio_annualized_vol_MR=np.std(portfolio_returns_MR)*np.sqrt(252)

        type_ptf = ["Minimum Risk","Sample"]
        PTF_S = ["Low Return/Risk","Mid Return/Risk","High Return/Risk"]
        own_vs_guided=["Proposition of Portfolios with Inputed Stocks","Investor's choices"]
        invest_choice=["Investment Percentage","Target Annual Return"]

        dico_ptf={}
        for i in range(len(PTF_S)):
            dico_ptf["Low Return/Risk"] =weights_S1
            dico_ptf["Mid Return/Risk"] =weights_S2
            dico_ptf["High Return/Risk"] =weights_S3

        if "typptf" not in st.session_state:
            st.session_state.typptf = "Minimum Risk"
        if "MT" not in st.session_state:
            st.session_state.MT = "Proposition of Portfolios with Inputed Stocks"
        if "chosen_ptf" not in st.session_state:
            st.session_state.chosen_ptf = "High Return/Risk"
        if "weightsorreturn" not in st.session_state:
            st.session_state.weightsorreturn = "Investment Percentage"
        if "weights" not in st.session_state:
            st.session_state.weights = None

        ## selection du type de ptf

        cols = st.columns(3)

        for i, management_type in enumerate(own_vs_guided):
            if cols[i].button(management_type):
                st.session_state.MT = management_type
        
            if st.session_state.MT:
                MT = st.session_state.MT
        
        if MT=="Proposition of Portfolios with Inputed Stocks":


            typptf = st.selectbox("**Choose the portfolio type**",type_ptf,index=type_ptf.index(st.session_state.typptf) if st.session_state.typptf else 0)
            
            st.session_state.typptf = typptf

            if typptf=="Minimum Risk":
                weights_plots_minv(typptf,weights_MV)
            else:
                cols = st.columns(3)

                for i, ptf_name in enumerate(PTF_S):
                    if cols[i].button(ptf_name):
                        st.session_state.chosen_ptf = ptf_name
                
                if st.session_state.chosen_ptf:
                    chosen_ptf = st.session_state.chosen_ptf
                    weights_plots_minv(chosen_ptf,dico_ptf[chosen_ptf])
        else:
            weightsorreturn = st.selectbox("**What do you wish to choose**",invest_choice,index=invest_choice.index(st.session_state.weightsorreturn) if st.session_state.weightsorreturn else 0)
            
            st.session_state.weightsorreturn = weightsorreturn
            if weightsorreturn=="Investment Percentage":
                with st.form("my_form"):
                    weights_vector = []  # list to store weights in input order

                    for item in assets_names:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(item)
                        with col2:
                            w = st.number_input(
                                f"Weight in {item} (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.01,
                                key=f"input_{item}"
                            )
                            weights_vector.append(w / 100)  # convert % to fraction
                    
                    submitt = st.form_submit_button("✅ Confirm Repartition")
                if submitt:
                    total_weight = sum(weights_vector)
                    if abs(total_weight - 1) > 1e-6:
                        st.error(f"⚠️ Invalid weights! Total = {total_weight:.2f} (should be 1)")
                    else:
                        st.session_state.weights = np.array(weights_vector)
                if st.session_state.weights is not None:
                    weights_plots_minv("Your Choice", st.session_state.weights)
            else:
                # --- ⚙️ SÉLECTION DU RENDEMENT CIBLE (UTILISE LES VALEURS EXISTANTES) ---


                st.markdown("Define your target annualized return")
                st.write(f"Enter a target between **{portfolio_annualized_return_MV*100:.2f}%** and **{portfolio_annualized_return_MR*100:.2f}%** per year.")
                target_return = st.number_input(
                    "Objectif de rendement annuel (%)",
                    min_value=portfolio_annualized_return_MV * 100,
                    max_value=portfolio_annualized_return_MR * 100,
                    value=(portfolio_annualized_return_MV*100 + portfolio_annualized_return_MR*100) / 2,
                    step=0.1,
                    format="%.2f"
                )

                # --- ⚙️ OPTIMISATION DU PORTEFEUILLE POUR LE TARGET RETURN ---
                if st.button("Launch Portfolio Optimization"):
        
                    target_return_daily = target_return / 100 / 252
                    opt_target = port_minvol_ro(mean, covariance_matrix,target_return_daily)
                        # ✅ Sauvegarde dans session_state
                    st.session_state["opt_target"] = opt_target
                    st.session_state["target_return_daily"] = target_return_daily


                # --- ⚙️ AFFICHAGE DES RÉSULTATS SI DÉJÀ OPTIMISÉ ---
                if "opt_target" in st.session_state:
                    opt_target = st.session_state["opt_target"]
                    target_return_daily = st.session_state["target_return_daily"]
                    
                    weights_plots_minv("Your Optimized",opt_target)
                    # 👉 ici tu gardes ton code d'affichage : graphiques, performance selon horizon, etc.
                else:
                    st.info("Choose a target return and click 'Launch Portfolio Optimization' to see the optimized portfolio.")