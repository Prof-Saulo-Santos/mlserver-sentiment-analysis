# Notebooks — Treinamento do Modelo

Este diretório contém o script de treinamento do modelo de análise de sentimentos.

## Como retreinar o modelo

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o script de treinamento:

```bash
python treinar_modelo.py
```

O modelo treinado será salvo automaticamente em:  
`../models/analisador-sentimentos/1.0/analisador-sentimentos-1.0.joblib`

## Sobre o modelo

- **Tipo:** Pipeline scikit-learn (`TfidfVectorizer` + `LogisticRegression`)
- **Categorias:** positivo, negativo, neutro
- **Dataset:** 105 amostras sintéticas em português (35 por categoria)
- **Acurácia:** ~86% no conjunto de teste

## Observações

- O dataset é sintético e criado para fins de demonstração
- Em um projeto real, seria recomendável usar datasets maiores como B2W Reviews ou TweetSentBR
- O pipeline salvo com `joblib` é compatível com o runtime `mlserver-sklearn` do MLServer
