import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import decomposition

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings("ignore")


def plot_value_counts(col_name, df):

    values_count = pd.DataFrame(df[col_name].dropna().value_counts())
    #print (values_count.shape)
    values_count.columns = ['count']
    # convert the index column into a regular column.
    values_count[col_name] = [str(i) for i in values_count.index]
    # add a column with the percentage of each data point to the sum of all data points.
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    # change the order of the columns.
    values_count = values_count.reindex([col_name, 'count', 'percent'], axis=1)
    values_count.reset_index(drop=True, inplace=True)
    return (values_count)

# entrée liste de transaction d'un utilisateur

df = pd.read_csv('fc_data.csv')
df.loc[:,'InvoiceDate'] = pd.to_datetime(df.InvoiceDate,errors='coerce')

valeurs_cust = plot_value_counts('CustomerID',df=df)
valeurs_cust = valeurs_cust.sort_values(by=['CustomerID'])
df_test = df[df.loc[:,'CustomerID'] == valeurs_cust.iloc[252,0]]

t = 1


#------------------------------------------------------------------------------------------

# fonction engendrant les caractéristiques necesaires à la ségmentation des modèles 1

# importer les informations de popularité des items
valeurs_items = pd.read_csv('valeurs_items.csv')

df = df_test
df.loc[:, 'InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')

ind_date = df.columns.get_loc('InvoiceDate')
df['DAY'] = df.iloc[:, ind_date].dt.dayofyear
df['HOUR'] = df.iloc[:, ind_date].dt.hour
df['MONTH'] = df.iloc[:, ind_date].dt.month

# repérer les indices

ind_price = df.columns.get_loc('UnitPrice')
ind_q = df.columns.get_loc('Quantity')
ind_cbis = df.columns.get_loc('Country_bis')

# ---------------------------------------- Associer les caractéristiques moyennes ------------------------------------
if (t > 0):
    # creer le dataset "dataset_clust"
    d = {'user': ['new_user'], 'nb_unit': [0], 'prix_unit': [0], 'montant': [0],
         'pays': [0], 'nb_produit_diff': [0], 'nb_commandes': [0], 'nb_produit_id': [0], 'frequence': [0],
         'jourmax': [0], 'heuremax': [0],
         'jourprob': [0], 'heureprob': [0], 'pop1': [0], 'pop2': [0]
         }
    dataset_clust = pd.DataFrame(data=d)

if (t == 0):
    # creer le dataset "dataset_clust"
    d = {'user': ['new_user'], 'nb_unit': [0], 'prix_unit': [0], 'montant': [0],
         'pays': [0], 'nb_produit_diff': [0], 'nb_commandes': [0], 'jourmax': [0], 'heuremax': [0],
         'jourprob': [0], 'heureprob': [0], 'pop1': [0], 'pop2': [0]
         }
    dataset_clust = pd.DataFrame(data=d)

# associer la valeur 1 lorsque le pays est uk et 0 sinon
df['Country_bis'][df.loc[:, 'Country_bis'] == 'UK'] = 1
df['Country_bis'][df.loc[:, 'Country_bis'] == 'Foreign'] = 0

# ----------------------------------------------------------------------------------

# remplir les colonnes

df_trans = df

dataset_clust.loc[0, 'nb_unit'] = np.mean(df_trans.iloc[:, ind_q])
dataset_clust.loc[0, 'prix_unit'] = np.mean(df_trans.iloc[:, ind_price])
dataset_clust.loc[0, 'montant'] = np.mean(df_trans.iloc[:, ind_price].multiply(df_trans.iloc[:, ind_q]))
dataset_clust.loc[0, 'pays'] = df_trans.iloc[0, ind_cbis]

valeurs_invoice = plot_value_counts('InvoiceNo', df=df)
dataset_clust.loc[0, 'nb_produit_diff'] = np.mean(valeurs_invoice.iloc[:, 1])
dataset_clust.loc[0, 'nb_commandes'] = valeurs_invoice.shape[0]

valeurs_identiques = plot_value_counts('StockCode', df=df)
if (t > 0):
    dataset_clust.loc[0, 'nb_produit_id'] = np.mean(valeurs_identiques.iloc[:, 1])

    dataset_clust.loc[0, 'frequence'] = (np.amax(df_trans.loc[:, 'InvoiceDate']) -
                                         np.amin(df_trans.loc[:, 'InvoiceDate'])) / df_trans.shape[0]

    dataset_clust.loc[0, 'frequence'] = dataset_clust.loc[0, 'frequence'].total_seconds()
    dataset_clust.loc[0, 'frequence'] = float(dataset_clust.loc[0, 'frequence'])

valeurs_day = plot_value_counts('DAY', df=df_trans)
valeurs_day.iloc[:, 0] = pd.to_numeric(valeurs_day.iloc[:, 0], errors='coerce').fillna(0, downcast='infer')

dataset_clust.loc[0, 'jourmax'] = valeurs_day.iloc[0, 0]
dataset_clust.loc[0, 'heuremax'] = valeurs_day.iloc[0, 1] / np.sum(valeurs_day.iloc[0, :])

valeurs_hour = plot_value_counts('HOUR', df=df_trans)
valeurs_hour.iloc[:, 0] = pd.to_numeric(valeurs_hour.iloc[:, 0], errors='coerce').fillna(0, downcast='infer')

dataset_clust.loc[0, 'jourprob'] = valeurs_hour.iloc[0, 0]
dataset_clust.loc[0, 'heureprob'] = valeurs_hour.iloc[0, 1] / np.sum(valeurs_hour.iloc[0, :])

# popularité
# considérer les items communs entre valeurs_items et valeurs_identiques
valeurs_pop = valeurs_items[(valeurs_items.loc[:, 'StockCode'].isin(valeurs_identiques.iloc[:, 0]))].reset_index(
    drop=True)
# le score de popularité utilisateur est la moyenne des scores items
dataset_clust.loc[0, 'pop1'] = np.mean(valeurs_pop.loc[:, 'pop1'])
dataset_clust.loc[0, 'pop2'] = np.mean(valeurs_pop.loc[:, 'pop2'])

df_manual2 = dataset_clust

# remplacer les "nan" par des 0

df_manual2 = dataset_clust.fillna(0)


# -------- pour 3 clusters -------------
# ouvrir les informations relatives au modèle
clf = joblib.load('model')

inter = pd.read_csv('cluster_annotN3.csv')

user = np.matrix(df_manual2.iloc[:, 1::])


labels_predict = clf.predict(user)

inter = inter[inter.loc[:, 'cluster_num'] == labels_predict[0]]

print ('##############################################################################################)')
print ("L'utilisateur dont vous avez rentré les informations est suceptible d'appartenir au cluster {}.".format(
    labels_predict[0]))
print ("caractéristiques du cluster {}: {}".format(labels_predict[0], inter.iloc[0, 2]))



