"""
Treinamento do Modelo: Analisador de Sentimentos em Português
==============================================================
Este script treina um classificador de sentimentos usando scikit-learn.
O pipeline combina TfidfVectorizer + LogisticRegression em um único objeto,
compatível com o MLServer (runtime mlserver-sklearn).

Categorias: positivo, negativo, neutro

Uso:
    python treinar_modelo.py

Saída:
    ../models/analisador-sentimentos/1.0/analisador-sentimentos-1.0.joblib
"""

import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# 1. Dataset sintético de sentimentos em português
# =============================================================================
# Em um projeto real, usaríamos um dataset como o B2W Reviews ou TweetSentBR.
# Aqui criamos um dataset curado para demonstração do MLServer.

dados_positivos = [
    "Adorei o produto, chegou rápido e em perfeito estado",
    "Excelente qualidade, superou minhas expectativas",
    "Muito bom, recomendo para todos",
    "Produto maravilhoso, estou muito satisfeito",
    "Ótima compra, valeu cada centavo",
    "Perfeito, exatamente como descrito no anúncio",
    "Amei, vou comprar de novo com certeza",
    "Fantástico, entrega rápida e produto impecável",
    "Sensacional, melhor compra que já fiz",
    "Incrível a qualidade deste produto",
    "Estou muito feliz com a compra",
    "Produto de primeira linha, nota dez",
    "Chegou antes do prazo e funcionando perfeitamente",
    "Qualidade excepcional pelo preço pago",
    "Simplesmente perfeito para o que eu precisava",
    "O melhor produto que já comprei nessa categoria",
    "Funciona muito bem, estou impressionado",
    "Recomendo fortemente, produto de altíssima qualidade",
    "Estou encantado com a qualidade do material",
    "Nota mil, superou todas as expectativas",
    "Produto chegou impecável, embalagem perfeita",
    "Gostei muito, atendeu todas as minhas necessidades",
    "Experiência de compra excelente do início ao fim",
    "Material resistente e bem acabado, parabéns",
    "Produto top, com certeza comprarei novamente",
    "A qualidade é surpreendente pelo valor cobrado",
    "Estou muito contente, funcionou de primeira",
    "Maravilha de produto, entrega e qualidade nota dez",
    "Sem defeitos, perfeito estado e ótimo funcionamento",
    "Custo benefício incrível, produto de primeira",
    "Obrigado pela rapidez na entrega e pelo produto excelente",
    "Surpreendente a qualidade, muito acima da média",
    "Produto lindo e funcional, adorei demais",
    "A melhor experiência de compra online que já tive",
    "Tudo perfeito, desde o atendimento até o produto final",
]

dados_negativos = [
    "Péssimo produto, não funciona direito",
    "Horrível, veio com defeito e quebrado",
    "Muito ruim, não recomendo para ninguém",
    "Produto de péssima qualidade, desperdicei dinheiro",
    "Terrível, não vale o preço cobrado",
    "Decepcionante, nada do que foi prometido",
    "Uma porcaria, parou de funcionar em dois dias",
    "Detestei, qualidade muito abaixo do esperado",
    "Pior compra que já fiz na vida",
    "Produto veio danificado e o vendedor não responde",
    "Não comprem, é furada total",
    "Muito insatisfeito com este produto",
    "Jogou dinheiro fora quem comprou isso",
    "Absolutamente horrível, devolvi imediatamente",
    "Produto falsificado e de péssima qualidade",
    "Uma vergonha, não funciona como deveria",
    "Totalmente decepcionado com a compra",
    "Material frágil e mal acabado, quebrável",
    "Não presta, parou de funcionar rapidamente",
    "Experiência horrível, nunca mais compro aqui",
    "Produto horrível, qualidade zero absoluto",
    "Péssimo acabamento, parece produto de segunda linha",
    "Veio totalmente diferente do anunciado",
    "Arrependido da compra, produto péssimo",
    "Lixo total, não serve para nada",
    "Qualidade deplorável, material de péssima procedência",
    "Não recomendo de jeito nenhum, pior produto possível",
    "Devolvi o produto pois era impossível usar",
    "Desperdício de tempo e dinheiro comprar isso",
    "Entrega atrasada e produto defeituoso",
    "Comprei e me arrependi amargamente",
    "Produto fraco e sem durabilidade nenhuma",
    "Muito abaixo do padrão esperado, decepcionante",
    "O pior produto que já vi nessa categoria",
    "Não funciona, é propaganda enganosa",
]

dados_neutros = [
    "O produto é razoável pelo preço",
    "Normal, nada de especial mas serve",
    "Produto ok, cumpre o que promete",
    "Mediano, poderia ser um pouco melhor",
    "Chegou no prazo, produto dentro do esperado",
    "Sem grandes surpresas, é o que é",
    "Produto básico, atende o necessário",
    "Nada demais, mas também não é ruim",
    "Regular, serve para o dia a dia",
    "Produto simples, sem muitos recursos",
    "Aceitável, mas existem opções melhores",
    "Dentro do esperado para o valor pago",
    "Nem bom nem ruim, apenas funcional",
    "Cumpre a função mas sem destaque",
    "Produto comum, sem nenhum diferencial",
    "Comprei por necessidade, atende o básico",
    "Indiferente, não tenho opinião forte",
    "Esperava mais mas também não decepcionou",
    "Produto que faz o mínimo necessário",
    "Sem reclamações mas também sem elogios",
    "Razoável para o preço que paguei",
    "É um produto padrão de mercado",
    "Funciona mas não tem nada especial",
    "Atende minimamente as necessidades básicas",
    "Comprei sem expectativas e recebi sem surpresas",
    "Produto mediano, qualidade aceitável",
    "Não posso reclamar, mas também não elogio",
    "Serve para quebrar um galho no dia a dia",
    "Produto genérico, cumpre o papel básico",
    "Satisfatório mas sem nenhum destaque especial",
    "Recebido em bom estado, funciona normalmente",
    "Produto convencional, nada fora do comum",
    "Qualidade intermediária, preço justo",
    "Básico porém funcional para uso diário",
    "Nada a declarar, produto dentro da média",
]

# Montar dataset
textos = dados_positivos + dados_negativos + dados_neutros
sentimentos = (
    ["positivo"] * len(dados_positivos)
    + ["negativo"] * len(dados_negativos)
    + ["neutro"] * len(dados_neutros)
)

print(f"Total de amostras: {len(textos)}")
print(f"  - positivo: {len(dados_positivos)}")
print(f"  - negativo: {len(dados_negativos)}")
print(f"  - neutro:   {len(dados_neutros)}")

# =============================================================================
# 2. Separação treino/teste
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    textos, sentimentos, test_size=0.2, random_state=42, stratify=sentimentos
)

print(f"\nTreino: {len(X_train)} amostras")
print(f"Teste:  {len(X_test)} amostras")

# =============================================================================
# 3. Construir Pipeline (TfidfVectorizer + LogisticRegression)
# =============================================================================
# O pipeline garante um único objeto com predict(), compatível com MLServer.
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classificador", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])

# =============================================================================
# 4. Treinamento
# =============================================================================
print("\nTreinando modelo...")
pipe.fit(X_train, y_train)

# =============================================================================
# 5. Avaliação
# =============================================================================
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcurácia: {accuracy:.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# =============================================================================
# 6. Teste com exemplos novos
# =============================================================================
exemplos = [
    "Produto incrível, amei demais!",
    "Uma porcaria total, não funciona",
    "É um produto normal, nada demais",
    "Estou muito feliz com essa compra maravilhosa",
    "Péssima qualidade, devolvi imediatamente",
]

print("Teste com exemplos novos:")
predicoes = pipe.predict(exemplos)
for texto, sentimento in zip(exemplos, predicoes):
    print(f"  [{sentimento:>8}] {texto}")

# =============================================================================
# 7. Salvar modelo com joblib
# =============================================================================
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "analisador-sentimentos", "1.0"
)
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "analisador-sentimentos-1.0.joblib")
joblib.dump(pipe, output_path)
print(f"\nModelo salvo em: {output_path}")
print("Treinamento concluído com sucesso!")
