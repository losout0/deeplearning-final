# 🧠 GPT2 - GQA: Geração de Texto com Arquitetura Customizada
Este projeto tem como objetivo treinar um modelo de linguagem baseado no **GPT-2** para geração de texto em português, utilizando obras de **Machado de Assis** como corpus principal. A arquitetura foi adaptada para tarefas de geração com avaliação qualitativa e checkpoints salvos ao longo do treinamento.

## 📁 Estrutura do projeto

```
deeplearning-final/
├── notebooks/              # Notebooks de experimentação e visualização
├── scripts/                # Scripts de ingestão e pré-processamento
├── src/
│   ├── gpt_2/              # Arquitetura customizada do modelo
│   └── utils/              # Funções auxiliares
├── train/                  # Loop de treinamento e avaliação
├── data/                   # Corpus limpo e dividido
├── checkpoints/            # Modelos salvos por época
├── logs/                   # Logs de treinamento
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## 🚀 Execução

O treinamento está sendo realizado em um notebook do Kaggle, aproveitando os recursos gratuitos de GPU. Para reproduzir:

1. Acesse o notebook no Kaggle
   - [GPT2 - MHA (Baseline)](https://www.kaggle.com/code/losout0/gpt-multihead)
   - [GPT2 - GQA](https://www.kaggle.com/code/losout0/gpt-gqa)
   - [Comparação GQA e MHA](https://www.kaggle.com/code/andrefelipealmeida/fork-of-compara-o-gpt-gqa-e-multihead)
2. Execute as células na ordem para:
   - Baixar e limpar os dados
   - Inicializar o modelo
   - Treinar e salvar checkpoints

## 📦 Dependências
Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
```
**Principais bibliotecas:**
- **torch** – Treinamento do modelo
- **tiktoken** – Tokenização eficiente
- **requests, tqdm** – Download e progresso
- **transformers** – Base para o GPT-2

## 📚 Dados
Os textos foram extraídos do [Projeto Gutenberg](https://www.gutenberg.org/) e processados para remover metadados, normalizar pontuação e dividir em parágrafos com tamanho mínimo.

## Referências
1. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

3. [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
