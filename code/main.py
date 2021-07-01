# %% 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

main_dir = os.path.dirname(os.getcwd())
data = os.path.join(main_dir, 'data', 'campeonato-brasileiro-estatisticas-full.csv')
stats = pd.read_csv(data, sep = ';')

# %%

stats.shape

# %%

stats.dropna(axis = 0, subset = ['Precisão de passe'], inplace = True)

# %%

stats.drop('ID', axis = 1, inplace = True)

# %%

to_num_passe = []
for x in stats['Precisão de passe']:
    to_num_passe.append(int(x.split('%')[0]))

# %% 

to_num_posse = []
for x in stats['Posse de bola']:
    to_num_posse.append(int(x.split('%')[0]))

# %%

stats['Posse de bola'] = to_num_posse
stats['Precisão de passe'] = to_num_passe
# até aqui o conjunto de dados foi preparado para análises futuras.
# a partir desse ponto será construido um modelo apenas para prever
# com base na quantidade de gols, se é mandante ou visitante

# %%

X, y = stats['Chutes a gol'], stats['Mandante']
print('X:', X.shape)
print('y:', y.shape)

# %%

seed = 42
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    random_state = seed,
                                                    test_size = test_size,
                                                    stratify = y
)

# %%

X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# %%

# aplicando o modelo de regressão logística
model= LogisticRegression(random_state = seed)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = model.score(X_test, y_test)

# %%

# gerando um baseline
DC = DummyClassifier()
DC.fit(X_train, y_train)
pred_DC = DC.predict(X_test)
acc_DC = DC.score(X_test, y_test)

# %%

# comparando resultados
print(f'Acurácia da Regressão Logística: {round(acc * 100, 2)}%')
print(f'Acurácia do Baseline: {round(acc_DC * 100, 2)}%')

# %%

d = {'cht_gol_teste': X_test.flatten(), 'alvo_previsto': pred}
comp = pd.DataFrame(d)

# %%

stats.head()

# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Chutes a Gol - Real', fontsize = 15)
sns.boxplot(ax = axes[0], data = stats[stats['Mandante'] == 0], y = 'Chutes a gol')
sns.boxplot(ax = axes[1], data = stats[stats['Mandante'] == 1], y = 'Chutes a gol')
axes[0].set_xlabel('Mandante', fontsize = 12)
axes[0].set_ylabel('Chutes a Gol', fontsize = 12)
axes[1].set_xlabel('Visitante', fontsize = 12)
axes[1].set_ylabel('Chutes a Gol', fontsize = 12)
plt.show()

# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Chutes a Gol - Previsto', fontsize = 15)
sns.boxplot(ax = axes[0], data = comp[comp['alvo_previsto'] == 0], y = 'cht_gol_teste')
sns.boxplot(ax = axes[1], data = comp[comp['alvo_previsto'] == 1], y = 'cht_gol_teste')
axes[0].set_xlabel('Mandante', fontsize = 12)
axes[0].set_ylabel('Chutes a Gol', fontsize = 12)
axes[1].set_xlabel('Visitante', fontsize = 12)
axes[1].set_ylabel('Chutes a Gol', fontsize = 12)
plt.show()
