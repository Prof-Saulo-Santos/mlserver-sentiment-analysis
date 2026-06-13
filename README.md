![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![MLServer](https://img.shields.io/badge/MLServer-1.7.1-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.1-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-%232496ED.svg?style=flat&logo=docker&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=flat&logo=nginx)
![MaterializeCSS](https://img.shields.io/badge/Materialize-CSS-pink.svg)

# Análise de Sentimentos: Model Serving com MLServer

_O objetivo é construir uma infraestrutura robusta de "Model Serving", garantindo que as predições do modelo estejam acessíveis, escaláveis e prontas para consumo por outras aplicações via APIs REST._

## Projeto

## Implementação Completa de um Servidor de Inferência com MLServer, Docker e Nginx para NLP.

Este manual reúne todas as fases de implementação do projeto, desde a criação e treinamento do pipeline no Scikit-Learn até a conteinerização e exposição do modelo usando proxy reverso. Ele foi desenhado para ser autossuficiente: basta clonar e executar o ambiente para interagir com a interface web que consome a API de inferência.

---

## 🏢 Simulação de Cenário Real & Valor de Negócio

Este projeto simula o ambiente produtivo de uma **plataforma de e-commerce ou rede social** que precisa analisar o feedback dos usuários em tempo real, mas precisa que essa inteligência esteja desacoplada dos sistemas principais.

**A Solução Fornecida:**
Implementamos uma infraestrutura de "Model Serving" dedicada que atua como o motor de inteligência artificial da empresa. Ela centraliza a execução dos modelos e fornece predições via API:

1.  **Desacoplamento Técnico:** O modelo de Machine Learning (Python/Scikit-Learn) roda em seu próprio ambiente (MLServer), sem sobrecarregar ou engessar o backend da aplicação principal.
2.  **Padronização de Comunicação:** Utilização do protocolo aberto **V2 Inference Protocol** (adotado pelo KServe/Triton), permitindo que qualquer sistema (Web, Mobile, Microserviços) faça requisições padronizadas.
3.  **Segurança e Escalabilidade:** Um proxy reverso (Nginx) gerencia as requisições de entrada, protegendo o MLServer e permitindo futura distribuição de carga.

---

## 🧠 Visão Geral da Arquitetura

O projeto implementa uma **Arquitetura de Serving Baseada em Containers**, com foco na separação de responsabilidades e facilidade de implantação.

A solução é composta pelos seguintes componentes:

- **Web Client (Frontend):** Uma interface em HTML/JS (Materialize CSS) que envia textos para análise e exibe o sentimento resultante.
- **Nginx (Proxy Reverso):** Escuta na porta `9080`. Ele serve os arquivos estáticos do Web Client na raiz (`/`) e atua como proxy reverso roteando chamadas de API (`/mlapi/*`) diretamente para o servidor de modelos.
- **MLServer (Motor de Inferência):** Roda na porta interna `8080`, carregando o modelo Scikit-Learn (`.joblib`) em memória e respondendo a predições seguindo o protocolo V2.

```
┌───────────────────────── Docker Compose ─────────────────────────┐
│                                                                  │
│  ┌───── nginx (:9080) ────┐      ┌──── mlserver (:8080) ──────┐  │
│  │                        │      │                            │  │
│  │  / ───────→ web client │      │  analisador-sentimentos    │  │
│  │                        │      │  (Pipeline Scikit-Learn)   │  │
│  │  /mlapi/* ─→ Proxy ────┼──────→  [POST /v2/models/.../infer]  │
│  └────────────────────────┘      └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 O Problema de Negócio: Por que Analisar Sentimentos?

Em um mercado competitivo, entender o que o cliente sente ao interagir com sua marca ou produto é essencial. 

1.  **Monitoramento de Satisfação:**
    *   **O Problema**: Ler milhares de reviews e feedbacks de clientes manualmente é lento e custoso.
    *   **A Solução (NLP)**: Um modelo classificador que lê o texto e categoriza automaticamente em Positivo, Negativo ou Neutro. Isso permite que equipes de atendimento foquem apenas nos casos críticos (Negativos).

2.  **Tomada de Decisão Ágil:**
    *   **O Problema**: Respostas lentas a crises de imagem.
    *   **A Solução (Real-time API)**: Ter o modelo servido via API permite que sistemas de monitoramento disparem alertas em tempo real se houver um pico repentino de comentários negativos.

**O Objetivo Final**: Extrair valor imediato de dados textuais não-estruturados, oferecendo inteligência acionável via uma API padronizada e confiável.

---

## 🤖 Enquadramento de Machine Learning

Aqui está o enquadramento do modelo construído para este projeto:

### Modelo de Classificação de Textos
*   **Tipo de Aprendizado:** **Aprendizado Supervisionado** (Supervised Learning).
*   **Tarefa:** **Classificação Multiclasse** (Multiclass Classification) e Processamento de Linguagem Natural (NLP).
*   **Explicação:** O modelo recebe textos em português pré-rotulados com seus sentimentos e aprende a categorizar novos textos nas classes: `positivo`, `negativo` ou `neutro`.
*   **Algoritmo Usado:** `TfidfVectorizer` (para extração de features textuais) concatenado com `LogisticRegression` (classificador linear) através de um **Pipeline** do Scikit-Learn.
*   **Performance:** ~86% de acurácia no dataset sintético de validação.

---

## 🏗️ Estrutura do Projeto

```text
.
├── compose.yaml              # Docker Compose (orquestra nginx + mlserver)
├── Dockerfile-mlserver       # Imagem isolada do motor de inferência
├── Dockerfile-nginx          # Imagem do proxy reverso e frontend
├── default                   # Configurações de roteamento do Nginx
├── requirements.txt          # Dependências do runtime (mlserver)
├── settings.json             # Configuração global (debug mode)
├── models/                   # Repositório de modelos (Model Registry local)
│   └── analisador-sentimentos/
│       └── 1.0/
│           ├── model-settings.json           # Contrato do modelo
│           └── analisador-sentimentos-1.0.joblib # Artefato treinado
├── webclient/
│   └── index.html            # Interface de demonstração do modelo
└── notebooks/
    ├── treinar_modelo.py     # Script para retreinar e gerar o artefato
    └── requirements.txt      # Dependências isoladas para treinamento
```

---

## 🚀 Execução Rápida

Para reproduzir este projeto em seu ambiente local e testar as inferências:

### 1. Iniciar a Infraestrutura (Containers)
```bash
# Sobe os containers de proxy e modelo em background
docker compose up --build -d
```

### 2. Testar pela Interface Visual
Abra o navegador e acesse:
[http://localhost:9080](http://localhost:9080)

<p align="center">
  <img src="analisador_sentimentos_1.jpg" width="45%" alt="Tela 1 - Analisador de Sentimentos">
  &nbsp;&nbsp;
  <img src="analisador_sentimentos_2.jpg" width="45%" alt="Tela 2 - Analisador de Sentimentos">
</p>

### 3. Testar pela API (V2 Protocol)
```bash
curl -X POST http://localhost:9080/mlapi/v2/models/analisador-sentimentos/versions/1.0/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "predict",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Adorei o produto, mas a entrega demorou."]
      }
    ]
  }'
```

### (Opcional) Retreinar o Modelo
```bash
cd notebooks
pip install -r requirements.txt
python treinar_modelo.py
```

---

## 📝 Autor

**Saulo Santos**

- GitHub: [https://github.com/Prof-Saulo-Santos](https://github.com/Prof-Saulo-Santos)
- LinkedIn: [https://www.linkedin.com/in/santossaulo/](https://www.linkedin.com/in/santossaulo/)
