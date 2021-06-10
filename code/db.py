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

df_full["Data"] = pd.to_datetime(df_full["Data"])

# %%

# visualização do número de registros por ano
df_full["Data"].dt.year.value_counts().sort_index().plot(kind = "bar")

# %%
