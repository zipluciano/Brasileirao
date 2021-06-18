# %%

# aplicação muito simples de um modelo preditivo, apenas para saber
# se determinado time é mandante ou visitante, dadas características de uma 
# partida de futebol

# %% 

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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

# %%

X, y = stats.iloc[:, 1:], stats['Mandante']
print('X:', X.shape)
print('y:', y.shape)

# %%

seed = 42
test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    random_state = seed,
                                                    test_size = test_size,
                                                    stratify = y
)

# %%

# aplicando o modelo Naive Bayes
mNB = MultinomialNB()
mNB.fit(X_train, y_train)
pred_mNB = mNB.predict(X_test)
acc_mNB = accuracy_score(y_test, pred_mNB)

# %%

# gerando um baseline
DC = DummyClassifier()
DC.fit(X_train, y_train)
pred_DC = DC.predict(X_test)
acc_DC = accuracy_score(y_test, pred_DC)

# %%

# comparando resultados
print(f'Acurácia do Naive Bayes: {round(acc_mNB * 100, 2)}%')
print(f'Acurácia do baseline: {round(acc_DC * 100, 2)}%')

# %%
