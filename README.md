# Kaggle-Medical-Appointment-No-Shows

Análise de dados e treinamento de um modelo de Machine Learning em um dataset da Kaggle que mapeia características dos pacientes ao comparecimento em uma consulta marcada. Dataset disponível em https://www.kaggle.com/joniarroba/noshowappointments

Características analisadas: gênero, dia em que a consulta foi marcada, dia da consulta, idade, local da consulta, escolaridade, hipertensão, diabetes, alcoolismo, deficiência (qualquer)

# O que foi feito
- Limpeza e Manipulação de dados para destacar padrões e padronizar base de dados
- Visualização de dados para ajudar na identificação de padrões entre as variáveis
- Testes estatísticos (Cramer's V) para embasar associações encontradas
- Seleção de variáveis com treinamento de modelos de Machine Learning

# Conlusão
Tanto as variáveis presentes no dataset quanto as criadas (via Feature Engineering) possuem baixo grau de associação com a variável alvo (comparecimento).
Apesar disso, um modelo de Árvore de Decisão (Decision Tree) foi criado e testado com uma parte separada dos dados.
O modelo criado performou acima de um modelo Dummy (onde a decisão é aleatória) nos dois conjuntos de dados (vistos e não vistos).
O modelo manteve uma Acurácia média de 59%.

# Tecnologias Utilizadas
- Pandas e Numpy (maniupalação de dados)
- Scikit-Learn (Machine Learning)
- Matplotlib e  Seaborn (Visualização de dados)
- Dython (Testes estatísticos)