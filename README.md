# Análise de Sentimento com PLN

Projeto prático para classificação supervisionada de sentimento (positivo/negativo) em avaliações de produtos em língua portuguesa, comparando Bag of Words (BoW) e TF-IDF associados à Regressão Logística, com validação cruzada e prevenção a vazamento de dados (*data leakage*).

## Estrutura do Projeto

```
sentimento-pln/
├── data/
│   ├── avaliacoes.csv
│   └── casos_dificeis.csv
├── src/
│   ├── __init__.py
│   ├── preprocessamento.py   # Normalização, tokenização e tratamento de stopwords
│   ├── modelos.py            # Pipelines e validação cruzada
│   └── avaliacao.py          # Métricas, matriz de confusão e análise de erros
├── resultados/               # Arquivos gerados pela execução do pipeline
├── main.py                   # Script principal que orquestra e executa tudo
├── RELATORIO.md              # Relatório detalhado das análises e respostas
├── requirements.txt          # Dependências do projeto
└── README.md
```

## Como Instalar e Executar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/marcodevweb/An-lise-de-Sentimento-com-PLN.git
   cd An-lise-de-Sentimento-com-PLN
   ```

2. **Crie e ative um ambiente virtual (opcional, recomendado):**
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/Mac:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o pipeline completo:**
   ```bash
   python main.py
   ```

Após a execução, todos os relatórios, matriz de confusão e predições estarão disponíveis na pasta `resultados/`.
Para conferir a análise crítica dos resultados, consulte o arquivo [RELATORIO.md](RELATORIO.md).
