# Detecção de estresse por meio da base de dados WESAD

Trabalho de Conclusão de Curso apresentado como requisito parcial à obtenção do título de Bacharel em Ciência da Computação na Área do Conhecimento de Ciências Exatas e Engenharias da Universidade de Caxias do Sul.

O fonte é dividido em 5 passos, que podem ser configurados e executados através do Service.py:
1. reader: realiza a leitura dos dados do arquivo pickle;
2. extractor: realiza a extração das características;
3. selector: realiza a seleção das características;
4. classifier: utilização dos classificadores;
5. shooter: utiliza a votação da maioria para ter uma decisão.

Além disso, é possível avaliar o desempenho dos classificadores através do evaluator.py

# Stress Detection with WESAD dataset

This project is divided in some steps:
1. reader: read the data from the pickle file;
2. extractor: extract the features from the data;
3. selector: select the best features from the features extracted;
4. classifier: classify the features with classifiers;
5. shooter: it's the algorithm to decide the answer.