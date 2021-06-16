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

drop_rodada = list(df_full['rodada'].unique()[34: 41])
df_full = df_full[~df_full['rodada'].isin(drop_rodada)]

# %%

df_full["data"] = pd.to_datetime(df_full["data"])
df_full['rodada'] = pd.to_numeric(df_full['rodada'])
df_full.reset_index(drop = True, inplace = True)
df_full.sort_values("data", ascending = True, inplace = True)

# %%

df_full["vencedor_casa"] = (df_full["mandante"] == df_full["vencedor"])
df_full["vencedor_fora"] = (df_full["visitante"] == df_full["vencedor"])
df_full["empate"] = (df_full["vencedor"] == "-")
df_full["estado_empate"] = (df_full["estado_vencedor"] == "-")
df_full["vencedor_casa"].replace({True: 1, False: 0}, inplace = True)
df_full["vencedor_fora"].replace({True: 1, False: 0}, inplace = True)
df_full["empate"].replace({True: 1, False: 0}, inplace = True)
df_full["estado_empate"].replace({True: 1, False: 0}, inplace = True)
df_full.drop("vencedor", inplace = True, axis = 1)

# %%

# mantendo um Data Frame para análise de dados(df_full) e criando
# outro para construção de algum modelo de machine learning(df_ml)
df_ml = df_full.copy()

# %%

df_ml['dia'] = df_ml['data'].dt.day
df_ml['mes'] = df_ml['data'].dt.month
df_ml['ano'] = df_ml['data'].dt.year
df_ml.drop('data', axis = 1, inplace = True)

# %%

# transformações da colunas de variáveis categóricas em numéricas
mandantes = pd.get_dummies(df_ml['mandante'], prefix = 'mandante')
visitantes = pd.get_dummies(df_ml['visitante'], prefix = 'visitante')
uf_mandante = pd.get_dummies(df_ml['estado_mandante'], prefix = 'uf_mandante')
uf_visitante = pd.get_dummies(df_ml['estado_visitante'], prefix = 'uf_visitante')
uf_vencedor = pd.get_dummies(df_ml['estado_vencedor'], prefix = 'uf_vencedor')
uf_empate = pd.get_dummies(df_ml['estado_empate'], prefix = 'uf_empate')

# %%

df_ml = pd.concat([df_ml, mandantes, visitantes], axis = 1)
drop_final = ['mandante', 
              'visitante',
              'estado_mandante',
              'estado_visitante',
              'estado_vencedor',
              'estado_empate',
              'id'
             ]
df_ml.drop(df_ml[drop_final], axis = 1, inplace = True)

# %%

# separando o data frame para aplicação do modelo
X, y = df_ml.drop('vencedor_casa', axis = 1), df_ml['vencedor_casa']

# %%

# aplicando uma árvore de decisão
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = seed)#,
                                                    #stratify = y
                                                    #)

model = DecisionTreeClassifier(
                              max_depth = 5
                              )
model.fit(X_train, y_train)
predict = model.predict(X_test)
acuracia = accuracy_score(y_test, predict)

# %%

# devido ao valor de acurácia obtido, é possível observar
# está ocorrendo overfitting
comparacao = pd.DataFrame({'y_test': y_test, 'predict': predict})
print('Acurácia: {:.3f}'.format(acuracia))
comparacao[comparacao.y_test != comparacao.predict]

# %%
