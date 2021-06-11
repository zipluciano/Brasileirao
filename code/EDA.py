# %%

# definindo os caminhos de pastas e arquivos
import os
import pandas as pd
import datetime as dt

code_dir = os.path.abspath(__file__)
code_dir = os.path.abspath(".")
data_dir = os.path.join(os.path.dirname(code_dir),
                        "data")
data_stats = os.path.join(data_dir, 
                    "campeonato-brasileiro-estatisticas-full.csv")
data_full = os.path.join(data_dir, 
                        "campeonato-brasileiro-full.csv")

# %%

# gerando os Data Frames
df_stats = pd.read_csv(data_stats, sep = ";")
df_full = pd.read_csv(data_full, sep = ";")

print(df_stats.shape)
print(df_full.shape)

# %%

df_full.info()

# %%

# removendo colunas desnecessárias para o propósito de análise
drop_col = ["Horário", "Dia", "Arena"]
df_full.drop(drop_col, axis = 1, inplace = True)

# %%

col_names = []
for x in df_full.columns:
    i = x.replace(" ", "_")
    col_names.append(i.lower()) 
df_full.columns = col_names

# %%

df_full["data"] = pd.to_datetime(df_full["data"])
df_full.sort_values("data", ascending = True, inplace = True)

# %%

# visualização do número de registros por ano
df_full["data"].dt.year.value_counts().sort_index().plot(kind = "bar")

# %%

df_full["vencedor_casa"] = (df_full["mandante"] == df_full["vencedor"])
df_full["vencedor_fora"] = (df_full["visitante"] == df_full["vencedor"])
df_full["empate"] = (df_full["vencedor"] == "-")
df_full["vencedor_casa"].replace({True: 1, False: 0}, inplace = True)
df_full["vencedor_fora"].replace({True: 1, False: 0}, inplace = True)
df_full["empate"].replace({True: 1, False: 0}, inplace = True)
df_full.drop("vencedor", inplace = True, axis = 1)

# %%

# mantendo um Data Frame para análise de dados(df_full) e criando
# outro para construção de algum modelo de machine learning(df_ml)
df_ml = df_full.copy()

# %%

drop_col_ml = ["mandante",
               "visitante",
               "data",
               "rodada", 
               "estado_mandante",
               "estado_visitante",
               "estado_vencedor"
              ]
df_ml.drop(drop_col_ml, axis = 1)

# %%

col_names_stats = []
for x in df_stats.columns:
    i = x.replace(" ", "_")
    col_names_stats.append(i.lower())  
df_stats.columns = col_names_stats

# %%

# ficaram poucos dados após a remoção de dados "NaN",
# o data set utilizado na análise será somente o df_ml
# posteriormente será feito uma análise com ambos
df_stats.dropna(subset = ["precisão_de_passe"])
