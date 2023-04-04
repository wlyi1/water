import numpy as np
import pandas as pd
import datetime as dt

def data_anom(df_con, tgl):
    result_DO = [x for x in df_con.loc[tgl,'DO']]
    result_pH = [x for x in df_con.loc[tgl,'pH']]
    result_NH = [x for x in df_con.loc[tgl,'NH4']]
    result_NO = [x for x in df_con.loc[tgl,'NO3']]
    result_COD = [x for x in df_con.loc[tgl,'BOD']]
    result_BOD = [x for x in df_con.loc[tgl,'COD']]
    ab_pH = sum(map(lambda x : x<6 and x>9, result_pH))
    ab_DO = sum(map(lambda x : x<3, result_DO))
    ab_NH4 = sum(map(lambda x : x>0.5, result_NH))
    ab_NO3 = sum(map(lambda x : x>20, result_NO))
    ab_BOD = sum(map(lambda x : x>12, result_BOD))
    ab_COD = sum(map(lambda x : x>100, result_COD))
    
    locals_stored = list(locals())
    list_var = dict()
    for i in locals_stored:
        list_var[i] = eval(i)
    return list_var

def ml(model, data_input):
    #training machine learning model
    status = model.predict(data_input)
    return status

def data(j, param, df, arr):
    arr1 = param.reshape(j,4,6)
    

    mean = []
    std_data = []
    std_grad = []
    max_data = []
    min_data = []
    
    for x in range(j):
        y = arr1[x].flatten()
        y_g = np.gradient(arr1[x].flatten())
        x_mean = np.mean(y)
        x_std = np.std(y)
        x_g_std = np.std(y_g)
        x_max = np.max(y)
        x_min = np.min(y)
        mean.append(x_mean)
        std_data.append(x_std)
        std_grad.append(x_g_std)
        max_data.append(x_max)
        min_data.append(x_min)

    df_ml = pd.DataFrame(list(zip(mean, max_data, min_data, std_data, std_grad)), columns = ['mean_data', 'max_data', 'min_data', 'std_data', 'std_grad'])
    arr_ml = df_ml.to_numpy()
        
    df_ml['TGL'] = df['logDate'].unique()
    #st.write(df_ml)
    df_ml['TGL'] = df_ml['TGL'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))

    df_ml = df_ml.set_index(pd.DatetimeIndex(df_ml['TGL']))
    df_ml.index = df_ml.index.strftime('%Y-%m-%d')
    
    return arr, df_ml, arr_ml
def chart(ylabel, xlabel, yvalues, xvalues, title=''):
    #create new graph
    
    fig = plt.figure(figsize = (10,7))
    plt.plot(xvalues, yvalues)
    plt.title(title, fontsize = 20, fontweight = 'bold')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return fig

def create_ml_data(feed, param):
    mean_data = list(feed[param].groupby(feed.index).mean().values)
    max_data = list(feed[param].groupby(feed.index).max().values)
    min_data = list(feed[param].groupby(feed.index).min().values)
    std_data = list(feed[param].groupby(feed.index).std().values)
    std_grad = list(feed[param].groupby(feed.index).apply(np.gradient).explode().groupby(feed.index).std().values)
    data = pd.DataFrame({'mean_data': mean_data, 'max_data': max_data, 'min_data': min_data, 'std_data' : std_data, 'std_grad' : std_grad})
    return data