# Ánalise de dados do Campeonato Brasileiro - Série A

<br /><div align="center"><img src="https://raw.githubusercontent.com/zipluciano/Brasileirao/master/docs/br_ao.jpg" alt="Texto alternativo caso imagem não esteja disponível" width="500px" height="300px"></div><br />

Criando um ambiente virtual com a versão mais recente do python, utilizando como gerenciador de pacotes o Anaconda
````
conda create --name env_campeonato python=3.
````

Ativando o ambiente virtual recém criado
````
conda activate env_campeonato
````

Adicionando os pacotes necessários para o desenvolvimento do projeto
````
pip install -r requirements.txt
````

Caso no painel interativo do VScode apareça "Kernel died with exit code 1", tente:
```
conda install ipykernel --update-deps
```
se não resolver, tente ainda:
```
conda install ipykernel --update-deps --force-reinstall
```
