import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import random

# Import data from csv
data_path = '/Users/cami/Library/CloudStorage/OneDrive-Raccoltecondivise-eNextGen/Nicolò Golinucci - Tesi Camilla/Model/representative days/data_Buccinasco/hourly_buccinsco.csv'
df = pd.read_csv(data_path, skiprows=10, delimiter=',', low_memory=False)
df = df.iloc[:-10]

# Select desired dataframes and create a the raw data DataFrame
raw = df[['time', 'P']].copy()
raw['P'] = pd.to_numeric(raw['P'], errors='coerce')

# Handle timestamp
raw['time'] = pd.to_datetime(raw['time'], format='%Y%m%d:%H%M')
raw['Year'] = raw['time'].dt.year
raw['Month'] = raw['time'].dt.month
raw['Day'] = raw['time'].dt.day
raw['Date'] = raw['time'].dt.strftime('%Y%m%d')
raw['Hour'] = raw['time'].dt.hour

seasons=['winter', 'mid-cold', 'mid-warm', 'summer']
# Remove 29th February
raw = raw[~((raw['Month'] == 2) & (raw['Day'] == 29))]
raw['doy'] = raw['time'].dt.dayofyear

# Define seasons
def get_season(day):
    if 1 <= day <= 74 or 320 <= day <= 366:
        return 'winter'
    elif 75 <= day <= 105 or 289 <= day <= 319:
        return 'mid-cold'
    elif 106 <= day <= 135 or 258 <= day <= 288:
        return 'mid-warm'
    elif 136 <= day <= 257:
        return 'summer'
    else:
        return 'unknown'
    
raw['Season'] = raw['doy'].apply(get_season)

#Typical year generation-------------------------------------------------------
# Monthly avarages
monthly_means = raw.groupby(['Month', 'Hour'])['P'].mean().reset_index()
typical_year = {}
for month in range(1, 13):
     month_data = raw[raw['Month'] == month]
    
     #Evaluate distance of monthly avarage for each year
     year_distances = {}
     for year in month_data['Year'].unique():
         year_data = month_data[month_data['Year'] == year]
         hourly_means = year_data.groupby('Hour')['P'].mean()
        
        # Allign index
         hourly_means = hourly_means.reindex(monthly_means[monthly_means['Month'] == month]['Hour'].values, fill_value=0)
         distance = ((hourly_means - monthly_means[monthly_means['Month'] == month]['P'].values) ** 2).sum()
         year_distances[year] = distance
    
     # Select year with min distance
     typical_year[month] = min(year_distances, key=year_distances.get)

#Unite
typical_year_data_list = []
for month, year in typical_year.items():
    month_data = raw[(raw['Year'] == year) & (raw['Month'] == month)]
    typical_year_data_list.append(month_data)

# Concatenazione dei dati in un unico DataFrame
typical_year_data = pd.concat(typical_year_data_list)
# Sort by timestamp
typical_year_data = typical_year_data.sort_values(by='time')
    

#Giorni di produzione minima e massima del TMY
# Calcolare la produzione giornaliera
daily_productionTMY = typical_year_data.groupby(['Date', 'Season'])['P'].sum().reset_index()

# Identificare i giorni di produzione minima e massima per ciascuna stagione
min_prod_dates = {}
max_prod_dates={}
for season in typical_year_data['Season'].unique():
    season_df = daily_productionTMY[daily_productionTMY['Season'] == season]
    min_prod_dates[season]=season_df.loc[season_df['P'].idxmin()]['Date']
    max_prod_dates[season]=season_df.loc[season_df['P'].idxmax()]['Date']

#Clustering--------------------------------------------------------------------
#Preparing for clustering
X=raw[['Date','Hour','P','Season']]
X['P'] = X['P'].fillna(0)
X['data'] = pd.to_datetime(X['Date'])

#Pivoting by date
pivot_data = raw.pivot_table(index='Date', columns='Hour', values='P')
pivot_data = pivot_data.fillna(0)

#CLUSTERING (from towardsdatascience.com)
K=6
daily_production = raw.groupby(['Date', 'Season'])['P'].sum().reset_index()
daily_production['Cluster'] =-1

centroids_df = pd.DataFrame()
Profiles=pd.DataFrame()
Profiles.index=range(1,25)
for s in seasons:
    # Pivot dei dati per ottenere un formato in cui ogni riga rappresenta un giorno e ogni colonna rappresenta un'ora
    X_season=X[X['Season']==s]
    X_pivot=X_season.pivot(index='Date', columns='Hour', values='P')
    X_pivot=X_pivot.fillna(0)

    # Riordinare le colonne per assicurarsi che siano in ordine da 0 a 23
    X_pivot = X_pivot[sorted(pivot_data.columns)]

    A=X_pivot.values.copy()

    sc = MinMaxScaler()
    A = sc.fit_transform(A)

    #Clustering
    kmeans = KMeans(n_clusters=K)
    cluster_found = kmeans.fit_predict(A)
    cluster_found_sr = pd.Series(cluster_found, name='cluster')
    X_pivot = X_pivot.set_index(cluster_found_sr, append=True )

    centroids = kmeans.cluster_centers_

    season_mask=daily_production['Season']==s
    daily_production.loc[season_mask,'Cluster'] = cluster_found

    # # Aggiungere i centroidi al DataFrame centroids_df
    # centroids_df_season = pd.DataFrame(centroids, columns=sorted(X_pivot.columns))
    # centroids_df_season['Season'] = s
    # centroids_df_season['Cluster'] = range(K)

    # # Aggiungere i centroidi al DataFrame centroids_df
    # centroids_df = pd.concat([centroids_df, centroids_df_season], ignore_index=True)

    fig, ax= plt.subplots(1,1, figsize=(18,10))
    color_list = ['blue','red','green','purple','orange','yellow']
    cluster_values = sorted(X_pivot.index.get_level_values('cluster').unique())

    for cluster, color in zip(cluster_values, color_list):
        X_pivot.xs(cluster, level=1).T.plot(
            ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
        )
        X_pivot.xs(cluster, level=1).median().plot(
            ax=ax, color=color, alpha=0.9, ls='--'
        )

    for i in range(K):
        Profiles[f'{s}_{i+1}']=X_pivot.xs(i, level=1).median().values.copy()
    
    Profiles[f'{s}_min']=raw[raw['Date'] == min_prod_dates[s]][['P']].values.copy()
    Profiles[f'{s}_max']=raw[raw['Date'] == max_prod_dates[s]][['P']].values.copy()
    plot_min = pd.Series(Profiles[f'{s}_min'])
    plot_min.index = range(24)  
    plot_max=Profiles[f'{s}_max']
    plot_max.set_axis(range(0,24),axis=0)
    ax.plot(range(24),plot_min,label='Min',color='lightblue')
    ax.plot(range(24),plot_max,label='Max',color='violet')
    ax.set_xticks(np.arange(1,25))
    ax.set_ylabel('W')
    ax.set_xlabel('hour')
    ax.title.set_text(f'Daily PV production profiles for {s}')
    plt.savefig(f'clustering_profiles/K6_season_{s}.png')

#Profiles.to_csv('PV_profiles.csv', index=False)

#How representative are the clusters?
def compute_coefficients(c, p, tot_prod,weekdays):    
    # Modello di ottimizzazione
    model = gp.Model("ottimizzazione_deltaC")
    #model.Params.OutputFlag = 0

    # Variabili decisionali deltaC (lunghezza 14, possono essere positive o negative)
    deltaC = model.addVars(14, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="deltaC")

    # Variabili ausiliarie per il valore assoluto di deltaC
    abs_deltaC = model.addVars(14, lb=0, ub=GRB.INFINITY, name="abs_deltaC")

    # Funzione obiettivo: minimizzare la somma dei prodotti tra |deltaC[i]| e c[i]
    objective = gp.quicksum(abs_deltaC[i] / c[i] for i in range(14) if c[i] != 0)
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraint 1: computed production within +-5% of the target
    model.addConstr(gp.quicksum((c[i] + deltaC[i]) * p[i] for i in range(14)) <= tot_prod*1.1, "constr_prod_tot_min")
    model.addConstr(gp.quicksum((c[i] + deltaC[i]) * p[i] for i in range(14)) >= tot_prod*0.9, "constr_prod_tot_max")

    # Constraint 2: sum(deltaC) = 0
    model.addConstr(gp.quicksum(deltaC[i] for i in range(14)) == 0, "constr_sum_deltaC")

    
    for i in range(14):
        # Constraint 3: deltaC[i] = 0 quando c[i] = 1
        if c[i] == 1:
            model.addConstr(deltaC[i] == 0, f"constr_deltaC_{i}")
        # Constraint abs_deltaC[i]=|deltaC[i]|
        model.addConstr(abs_deltaC[i] >= deltaC[i], f"abs_deltaC_pos_{i}")
        model.addConstr(abs_deltaC[i] >= -deltaC[i], f"abs_deltaC_neg_{i}")
        # Vincolo per garantire che abs_deltaC[i] non sia più del 20% di c[i]
        model.addConstr(abs_deltaC[i] <= 0.3 * c[i], f"vincolo_max_15_percent_{i}")
        #No negative coefficients
        model.addConstr(c[i] + deltaC[i] >= 0, f"vincolo_c_deltaC_pos_{i}")

    # Vincolo: somma dei coefficienti da 1 a 5 e da 8 a 12 deve essere entro ±1% del target_ratio
    somma_coeff = gp.quicksum(c[i] + deltaC[i] for i in range(5)) + gp.quicksum(c[i] + deltaC[i] for i in range(7, 12))
    model.addConstr(somma_coeff >= 0.96 * weekdays, "vincolo_somma_min")
    model.addConstr(somma_coeff <= 1.04 * weekdays, "vincolo_somma_max")
    
    # Risoluzione del problema
    model.optimize()

    # Output dei risultati
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        for i in range(14):
            print(f"deltaC[{i}] = {deltaC[i].x}")
    else:
        print("No solution found.")

    # Extract the optimized values of deltaC
    deltaC_values = np.array([deltaC[i].x for i in range(14)])

    # Compute new_c
    new_c = c + deltaC_values
    return new_c

#Summer has a different setup of days
def compute_coefficients_summer(c, p, tot_prod,weekdays):    
    # Modello di ottimizzazione
    model = gp.Model("ottimizzazione_deltaC")
    #model.Params.OutputFlag = 0

    # Variabili decisionali deltaC (lunghezza 14, possono essere positive o negative)
    deltaC = model.addVars(14, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="deltaC")

    # Variabili ausiliarie per il valore assoluto di deltaC
    abs_deltaC = model.addVars(14, lb=0, ub=GRB.INFINITY, name="abs_deltaC")

    # Funzione obiettivo: minimizzare la somma dei prodotti tra |deltaC[i]| e c[i]
    objective = gp.quicksum(abs_deltaC[i] / c[i] for i in range(14) if c[i] != 0)
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraint 1: computed production within +-5% of the target
    model.addConstr(gp.quicksum((c[i] + deltaC[i]) * p[i] for i in range(14)) <= tot_prod*1.05, "constr_prod_tot_min")
    model.addConstr(gp.quicksum((c[i] + deltaC[i]) * p[i] for i in range(14)) >= tot_prod*0.95, "constr_prod_tot_max")

    # Constraint 2: sum(deltaC) = 0
    model.addConstr(gp.quicksum(deltaC[i] for i in range(14)) == 0, "constr_sum_deltaC")

    
    for i in range(14):
        # Constraint 3: deltaC[i] = 0 quando c[i] = 1
        if c[i] == 1:
            model.addConstr(deltaC[i] == 0, f"constr_deltaC_{i}")
        # Constraint abs_deltaC[i]=|deltaC[i]|
        model.addConstr(abs_deltaC[i] >= deltaC[i], f"abs_deltaC_pos_{i}")
        model.addConstr(abs_deltaC[i] >= -deltaC[i], f"abs_deltaC_neg_{i}")
        # Vincolo per garantire che abs_deltaC[i] non sia più del 20% di c[i]
        model.addConstr(abs_deltaC[i] <= 0.3 * c[i], f"vincolo_max_15_percent_{i}")
        #No negative coefficients
        model.addConstr(c[i] + deltaC[i] >= 0, f"vincolo_c_deltaC_pos_{i}")

    # Vincolo: somma dei coefficienti weekdays deve essere entro ±5% del target_ratio
    somma_coeff = gp.quicksum(c[i] + deltaC[i] for i in range(5)) + gp.quicksum(c[i] + deltaC[i] for i in range(7, 11))
    model.addConstr(somma_coeff >= 0.95 * weekdays, "vincolo_somma_min")
    model.addConstr(somma_coeff <= 1.05 * weekdays, "vincolo_somma_max")
    
    # Risoluzione del problema
    model.optimize()

    # Output dei risultati
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        for i in range(14):
            print(f"deltaC[{i}] = {deltaC[i].x}")
    else:
        print("No solution found.")

    # Extract the optimized values of deltaC
    deltaC_values = np.array([deltaC[i].x for i in range(14)])

    # Compute new_c
    new_c = c + deltaC_values
    return new_c


# Seasonal parameters: days per season -2 per min and max
season_days = {
    'winter': 118,
    'mid-cold': 60,
    'mid-warm': 58,
    'summer': 121
}
#Workdays per season
workdays = {
    'winter': 74,
    'mid-cold': 37,
    'mid-warm': 38,
    'summer': 67
}
#PV production
tot_prod = {
    'winter': 297.29,
    'mid-cold': 201.69,
    'mid-warm': 245.22,
    'summer': 590.44
}

df_index=['w1','w2','w3','w4','w5','h1','h2','w6','w7','w8','w9','w10','h3','h4']
#May be needed to run multiple times to get the right coefficients
def profiles_and_coefficients(daily_production, seasons, Profiles, season_days, tot_prod, workdays):
    random_sequence=pd.DataFrame()
    coeff_sequence=pd.DataFrame()
    coefficients=pd.DataFrame(columns=seasons,index=df_index)
    daily_sum=pd.DataFrame(columns=seasons,index=df_index)
    new_coefficients=pd.DataFrame(columns=seasons,index=df_index)
    for s in seasons:
        #Report the distrubition coefficients of clusters
        season_data = daily_production[daily_production['Season'] == s]
        cluster_counts = season_data['Cluster'].value_counts(normalize=True)
        
        #Build sequence of daus and coefficients as found by clusteting
        days_sequence=pd.DataFrame()
        for i in range(1,7):
            days_sequence[f'day{i}']=Profiles[f'{s}_{i}']
            coeff_sequence[f'day{i}']=[cluster_counts[i-1]*season_days[s]/2]
        for i in range(7,13):
            days_sequence[f'day{i}']=Profiles[f'{s}_{i-6}']
            coeff_sequence[f'day{i}']=[cluster_counts[i-7]*season_days[s]/2]      
        days_sequence['day13']=Profiles[f'{s}_max']    
        days_sequence['day14']=Profiles[f'{s}_min']       
        coeff_sequence['day13']=coeff_sequence['day14']=[1]
        
        order = list(range(1,15))
        random.shuffle(order)

        #Build the sequence of shuffled days, dividing weekdays and holidays
        for i in range(14):
            random_sequence[f'{s}{i+1}_{df_index[i]}']=days_sequence[f'day{order[i]}']
            coefficients.loc[df_index[i],s]=coeff_sequence[f'day{order[i]}'][0]
            daily_sum.loc[df_index[i],s]=random_sequence[f'{s}{i+1}_{df_index[i]}'].sum()
        
        #Fix coefficients (may need multiple runs to obtain the right randomized base)
        season_prod_computed=sum(coefficients[s]*daily_sum[s])/1000
        coefficients[s]=pd.to_numeric(coefficients[s],errors='coerce') 
        if s != 'summer':
            print(f'--->Fixing coefficients for {s}')
            new_coefficients[s]=compute_coefficients(coefficients[s], daily_sum[s], tot_prod[s]*1000,workdays[s])
            season_prod_computed=sum(new_coefficients[s]*daily_sum[s])/1000
        else:
            print('--->Fixing coefficients for summer')
            new_coefficients[s]=compute_coefficients_summer(coefficients['summer'], daily_sum['summer'], tot_prod['summer']*1000,workdays['summer'])
            season_prod_computed=sum(new_coefficients['summer']*daily_sum['summer'])/1000

    return new_coefficients, random_sequence

#Create days per type dataframe
days_per_type=pd.DataFrame(columns=['days_names','values'],index=range(57))
for i in range(57):
    if i<9:
        days_per_type.loc[i,'days_names']=f'winter0{i+1}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i],'winter']
    elif i<14:
        days_per_type.loc[i,'days_names']=f'winter{i+1}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i],'winter']
    elif i<23:
        days_per_type.loc[i,'days_names']=f'mid-cold0{i-13}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-14],'mid-cold']
    elif i<28:
        days_per_type.loc[i,'days_names']=f'mid-cold{i-13}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-14],'mid-cold']
    elif i<37:
        days_per_type.loc[i,'days_names']=f'mid-warm0{i-27}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-28],'mid-warm']
    elif i<42:
        days_per_type.loc[i,'days_names']=f'mid-warm{i-27}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-28],'mid-warm']
    elif i<51:
        days_per_type.loc[i,'days_names']=f'summer0{i-41}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-42],'summer']
    elif i<56:
        days_per_type.loc[i,'days_names']=f'summer{i-41}'
        days_per_type.loc[i,'values']=new_coefficients.loc[df_index[i-42],'summer']
    elif i==56:
        days_per_type.loc[i,'days_names']='peak'
        days_per_type.loc[i,'values']=0
days_per_type.to_excel('Data_input_DPT.xlsx',index=False)

#Export profiles as capacity factors for PV
cf=pd.DataFrame(columns=['time_h','values'])
j=0
season_short=['wi','mc','mw','su']
for s in range(4):
    for i in range(1,15):
        profile_column=random_sequence.loc[:, random_sequence.columns.str.contains(f'{seasons_list[s]}{i}_')]
        if i<10:
            for h in range(1,25):
                if h<10:
                    cf.loc[j,'time_h']=f'h0{h}_{season_short[s]}0{i}'
                    cf.loc[j,'values']=profile_column.iloc[h-1,0]/1000
                    j+=1
                else:
                    cf.loc[j,'time_h']=f'h{h}_{season_short[s]}0{i}'
                    cf.loc[j,'values']=profile_column.iloc[h-1,0]/1000
                    j+=1
        else:
            for h in range(1,25):
                if h<10:
                    cf.loc[j,'time_h']=f'h0{h}_{season_short[s]}{i}'
                    cf.loc[j,'values']=profile_column.iloc[h-1,0]/1000
                    j+=1
                else:
                    cf.loc[j,'time_h']=f'h{h}_{season_short[s]}{i}'
                    cf.loc[j,'values']=profile_column.iloc[h-1,0]/1000
                    j+=1
cf.loc[1344,'time_h']='peak'
cf.loc[1344,'values']=0
cf.to_excel('Data_input_cf_PV.xlsx',index=False)

#Perparing demand profiles Y ---------------------------------------------------
DPT=new_coefficients.copy()
days_per_season=pd.DataFrame(index=seasons)
days_per_season['tot'] = [120, 62, 60, 123]
days_per_season['workdays'] = [74, 37, 38, 67]
days_per_season['weekend'] = [46, 25, 22, 56]
for s in seasons:
    days_per_season.at[s, 'actual_workdays'] =sum(DPT.loc[[0,1,2,3,4,7,8,9,10,11], s].values)
    days_per_season.at[s, 'actual_holidays'] =sum(DPT.loc[[5, 6, 12, 13], s].values)

demand_per_season=pd.DataFrame(index=['winter', 'mid-cold', 'mid-warm', 'summer'])
demand_per_season['EE'] = [1951.67,	845.36,	758.28,	1270.75]
demand_per_season['Heat'] = [13306.36,4654.88,0,0]
demand_per_season['HW']=[947.26,456.39,	405.68,	643.00]
demand_per_season['Cook']=[46.40,22.98,23.86,36.77]

#--Demand profile for EE--
#Reference from literature
base_EE=pd.DataFrame(index=range(1,25),columns=['workday','holiday'])
base_EE['workday']=[0.19, 0.15, 0.13, 0.12, 0.125, 0.2, 0.31, 0.385, 0.29, 0.41, 0.4, 0.35, 0.33, 0.33, 0.34, 0.35, 0.46, 0.54, 0.61, 0.59, 0.56, 0.505, 0.42, 0.28]
base_EE['holiday']=[0.195, 0.15, 0.13, 0.13, 0.125, 0.18, 0.18, 0.25, 0.39, 0.51, 0.55, 0.43, 0.47, 0.425, 0.405, 0.395, 0.48, 0.5, 0.515, 0.57, 0.505, 0.48, 0.39, 0.285]
seasons2=['winter_work','winter_holi', 'mid-cold_work','mid-cold_holi', 'mid-warm_work','mid-warm_holi', 'summer_work','summer_holi']
demand_EE=pd.DataFrame(index=range(1,25),columns=seasons2)
#Building the daily demand and then profiles
EE_daily=pd.DataFrame(index=seasons,columns=['workday','holiday'])
for s in seasons:
    coeff_holi_work=1.03164
    EE_daily.at[s,'workday'] =demand_per_season.at[s,'EE']/(days_per_season.at[s,'actual_holidays']*coeff_holi_work+days_per_season.at[s,'actual_workdays'])
    EE_daily.at[s,'holiday'] =EE_daily.at[s,'workday']*coeff_holi_work
    demand_EE[f'{s}_work']=base_EE['workday']*EE_daily.at[s,'workday']/base_EE['workday'].sum()
    demand_EE[f'{s}_holi']=base_EE['holiday']*EE_daily.at[s,'holiday']/base_EE['holiday'].sum()

#--Demand profile for Heat--
#Reference profile
base_Heat=pd.DataFrame(index=range(1,25),columns=['workday_wi','holiday_wi','workday_mc','holiday_mc'])
base_Heat['workday_wi']=[0, 0, 0, 0, 0, 0, 0.05, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.06, 0.055, 0.055, 0.055, 0.055, 0.055, 0.06, 0.065, 0.05, 0, 0]
base_Heat['holiday_wi']=[0, 0, 0, 0, 0, 0, 0, 0.05, 0.09, 0.09, 0.08, 0.075, 0.06, 0.06, 0.055, 0.055, 0.055, 0.055, 0.06, 0.065, 0.07, 0.05, 0.03, 0]
base_Heat['workday_mc']=[0, 0, 0, 0, 0, 0, 0.05, 0.09, 0.09, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.07, 0.08, 0.09, 0.09, 0.09, 0.08, 0, 0]
base_Heat['holiday_mc']=[0, 0, 0, 0, 0, 0, 0, 0.04, 0.09, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.07, 0.08, 0.09, 0.08, 0.08, 0.03, 0]

demand_Heat=pd.DataFrame(index=range(1,25),columns=seasons2)
#Building the daily demand and then profiles
Heat_daily=pd.DataFrame(index=seasons,columns=['workday','holiday'])

coeff_holi_work=1.15
#Winter
Heat_daily.at['winter','workday'] =demand_per_season.at['winter','Heat']/(days_per_season.at['winter','actual_holidays']*coeff_holi_work+days_per_season.at['winter','actual_workdays'])
Heat_daily.at['winter','holiday'] =Heat_daily.at['winter','workday']*coeff_holi_work
demand_Heat['winter_work']=base_Heat['workday_wi']*Heat_daily.at['winter','workday']
demand_Heat['winter_holi']=base_Heat['holiday_wi']*Heat_daily.at['winter','holiday']
#Mid-cold
Heat_daily.at['mid-cold','workday'] =demand_per_season.at['mid-cold','Heat']/(days_per_season.at['mid-cold','actual_holidays']*coeff_holi_work+days_per_season.at['mid-cold','actual_workdays'])
Heat_daily.at['mid-cold','holiday'] =Heat_daily.at['mid-cold','workday']*coeff_holi_work
demand_Heat['mid-cold_work']=base_Heat['workday_mc']*Heat_daily.at['mid-cold','workday']
demand_Heat['mid-cold_holi']=base_Heat['holiday_mc']*Heat_daily.at['mid-cold','holiday']
#Mid-warm & Summer = no heat
for s in ['mid-warm','summer']:
    demand_Heat[f'{s}_work']=0
    demand_Heat[f'{s}_holi']=0

#--Demand profile for Hot Water--
#Reference from literature
diz={'winter_work': [0, 0, 0, 0, 0, 0, 0.05, 0.1275, 0.1175, 0.05, 0.01, 0.01, 0.01, 0.04, 0, 0, 0.01, 0.09, 0.09, 0.145, 0.105, 0.1, 0.045, 0],
    'winter_holi': [0, 0, 0, 0, 0, 0, 0, 0.01, 0.0825, 0.1075, 0.0775, 0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.0825, 0.1025, 0.105, 0.1, 0.0925, 0.09, 0],
    'mid-cold_work': [0, 0, 0, 0, 0, 0, 0.05, 0.1275, 0.1175, 0.05, 0.01, 0.01, 0.01, 0.04, 0, 0, 0.01, 0.09, 0.09, 0.145, 0.105, 0.1, 0.045, 0],
    'mid-cold_holi': [0, 0, 0, 0, 0, 0, 0, 0.01, 0.0825, 0.1075, 0.0775, 0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.0825, 0.1025, 0.105, 0.1, 0.0925, 0.09, 0],
    'mid-warm_work': [0, 0, 0, 0, 0, 0, 0.05, 0.1275, 0.1175, 0.05, 0.01, 0.01, 0.01, 0.04, 0, 0, 0.01, 0.09, 0.09, 0.145, 0.105, 0.1, 0.045, 0],
    'mid-warm_holi': [0, 0, 0, 0, 0, 0, 0, 0.01, 0.0825, 0.1075, 0.0775, 0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.0825, 0.1025, 0.105, 0.1, 0.0925, 0.09, 0],
    'summer_work': [0, 0, 0, 0, 0, 0, 0.05, 0.1275, 0.1175, 0.05, 0.01, 0.01, 0.01, 0.04, 0, 0, 0.01, 0.09, 0.09, 0.145, 0.105, 0.1, 0.045, 0],
    'summer_holi': [0, 0, 0, 0, 0, 0, 0, 0.01, 0.0825, 0.1075, 0.0775, 0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.0825, 0.1025, 0.105, 0.1, 0.0925, 0.09, 0]}
base_HW=pd.DataFrame(diz)
base_HW.index=range(1,25)

demand_HW=pd.DataFrame(index=range(1,25),columns=seasons2)
#Building the daily demand and then profiles
HW_daily=pd.DataFrame(index=seasons)
for s in seasons:
    HW_daily.at[s,'values'] =demand_per_season.at[s,'HW']/(days_per_season.at[s,'tot'])
    demand_HW[f'{s}_work']=base_HW[f'{s}_work']*HW_daily.at[s,'values']
    demand_HW[f'{s}_holi']=base_HW[f'{s}_holi']*HW_daily.at[s,'values']

#--Demand profile for cooking--
#Reference from literature
diz={
    'winter_work': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
    'winter_holi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.1, 0.5, 0.5, 0, 0, 0],
    'mid-cold_work': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
    'mid-cold_holi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
    'mid-warm_work': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 0, 0, 0, 0, 0, 0.4, 0.4, 0, 0, 0],
    'mid-warm_holi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.4, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
    'summer_work': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0, 0, 0, 0, 0, 0.3, 0.3, 0, 0, 0],
    'summer_holi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0, 0, 0, 0, 0, 0.3, 0.3, 0, 0, 0]}
base_cook=pd.DataFrame(diz)
base_cook.index=range(1,25)

demand_cook=pd.DataFrame(index=range(1,25),columns=seasons2)
#Building the daily demand and then profiles
cook_daily=pd.DataFrame(index=seasons)
for s in seasons:
    coeff_holi_work=1.8
    cook_daily.at[s,'workday'] = demand_per_season.at[s,'Cook']/(days_per_season.at[s,'actual_holidays']*coeff_holi_work+days_per_season.at[s,'actual_workdays'])
    cook_daily.at[s,'holiday'] = cook_daily.at[s,'workday']*coeff_holi_work
    demand_cook[f'{s}_work']=base_cook[f'{s}_work']*cook_daily.at[s,'workday']/base_cook[f'{s}_work'].sum()
    demand_cook[f'{s}_holi']=base_cook[f'{s}_holi']*cook_daily.at[s,'holiday']/base_cook[f'{s}_holi'].sum()

#--Demand profile for Transport --
transport_U=pd.DataFrame(index=range(1,15),columns=seasons)
transport_U['winter'] = [20, 20, 20, 20, 20, 10, 10, 20, 20, 20, 20, 20, 10, 10]
transport_U['mid-cold'] = [25, 25, 25, 25, 25, 10, 10, 25, 25, 25, 25, 25, 10, 10]
transport_U['mid-warm'] = [25, 25, 25, 25, 25, 4, 4, 25, 25, 25, 25, 25, 4, 4]
transport_U['summer'] = [30, 30, 30, 30, 30, 34, 34, 30, 30, 30, 30, 34, 34, 34]
transport_M=pd.DataFrame(index=range(1,15),columns=seasons)
transport_M['winter'] = [0, 0, 0, 0, 0, 30, 30, 0, 0, 0, 0, 0, 30, 30]
transport_M['mid-cold'] = [0, 0, 0, 0, 0, 40, 40, 0, 0, 0, 0, 0, 40, 40]
transport_M['mid-warm'] = [0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 100, 100]
transport_M['summer'] = [0, 0, 0, 0, 0, 60, 60, 0, 0, 0, 0, 0, 60, 60]

base_U=pd.DataFrame(index=range(1,25),columns=seasons2) #rifare
base_U['winter_work'] = [0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0]
base_U['winter_holi'] = [0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,3,0,0,0,3]
base_U['mid-cold_work'] = [0,0,0,0,0,0,0,0,10,0,0,0,5,0,0,0,0,0,10,0,0,0,0,0]
base_U['mid-cold_holi'] = [0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,3,0,0,0,3]
base_U['mid-warm_work'] = [0,0,0,0,0,0,0,0,10,0,0,0,5,0,0,0,0,0,10,0,0,0,0,0]
base_U['mid-warm_holi'] = [0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0]
base_U['summer_work'] = [0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,10,10,0,0,0,0]
base_U['summer_holi'] =[0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,15,0,0,0,15]


base_M=pd.DataFrame(index=range(1,25),columns=seasons2) #rifare
base_M['winter_work'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['winter_holi'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['mid-cold_work'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['mid-cold_holi'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['mid-warm_work'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['mid-warm_holi'] = [0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0]
base_M['summer_work'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
base_M['summer_holi'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0]

demand_trU=pd.DataFrame(index=range(1,25),columns=seasons2)
demand_trM=pd.DataFrame(index=range(1,25),columns=seasons2)

tot_km=0
for s in seasons:
    tot_km+=(transport_U[s]*DPT[s]).sum()+(transport_U[s]*DPT[s]).sum()

coeff=10000/tot_km

for s in seasons2:
    demand_trU[s]=base_U[s]*coeff
    demand_trM[s]=base_M[s]*coeff

#Costruzione colonna Y
#xls = pd.ExcelFile(file_path)
def extend_Y(demand,name):
    Y=pd.DataFrame()
    hours = ['h01','h02','h03','h04','h05','h06','h07','h08','h09','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24']
    days=['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
    short_seasons=['wi','mc','mw','su']
    j=0
    for s in range(4):
        for i in days:
            if i in ['01','02','03','04','05','08','09','10','11','12']:
                for h in range(24):
                    Y.loc[j,'n_names']=name
                    Y.loc[j,'th_names']=f'{hours[h]}_{short_seasons[s]}{i}'
                    Y.loc[j,'values']=demand.loc[h+1,f'{seasons[s]}_work']
                    j+=1
            elif i in ['06','07','13','14']:
                for h in range(24):
                    Y.loc[j,'n_names']=name
                    Y.loc[j,'th_names']=f'{hours[h]}_{short_seasons[s]}{i}'
                    Y.loc[j,'values']=demand.loc[h+1,f'{seasons[s]}_holi']
                    j+=1
    Y.loc[j,'n_names']=name
    Y.loc[j,'th_names']='peak'
    Y.loc[j,'values']=0
    j+=1
    return Y

Y=pd.DataFrame()
Y=extend_Y(demand_EE,'EE')
Ynull=Y.copy()
Ynull['values']=0
Y = pd.concat([Y, extend_Y(demand_HW,'Hot water')], ignore_index=True)
Y = pd.concat([Y, extend_Y(demand_Heat,'Heat')], ignore_index=True)
Y = pd.concat([Y, extend_Y(demand_cook,'Cooking')], ignore_index=True)
Ynull['n_names']='Natural Gas'
Y = pd.concat([Y, Ynull], ignore_index=True)
Ynull['n_names']='BEV battery discharge'
Y = pd.concat([Y, Ynull], ignore_index=True)
Y = pd.concat([Y, extend_Y(demand_trU,'Transport urban')], ignore_index=True)
Y = pd.concat([Y, extend_Y(demand_trM,'Transport motorway')], ignore_index=True)

# Write new file
with pd.ExcelWriter('Data_input_Y.xlsx', engine='openpyxl') as writer:
    Y.to_excel(writer, index=False,header=False)