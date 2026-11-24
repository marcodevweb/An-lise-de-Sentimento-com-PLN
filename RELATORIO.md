# RELATORIO — Análise de Sentimento com PLN

**Nome:** Marco Dev
**Repositório:** https://github.com/marcodevweb/An-lise-de-Sentimento-com-PLN


## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420 | 0.8564 ± 0.0926 | 0.8563 ± 0.0926 |
| B | mantidas | TF-IDF | 420 | 0.8462 ± 0.0843 | 0.8459 ± 0.0844 |
| C | removidas | BoW | 401 | 0.8564 ± 0.0661 | 0.8563 ± 0.0661 |
| D | removidas | TF-IDF | 401 | 0.8359 ± 0.0836 | 0.8354 ± 0.0842 |

**Configuração escolhida:** A (BoW com stopwords mantidas) **Acurácia no teste:** 0.8154 **F1 no teste:** 0.8150

---

## 1. Pré-processamento

No pré-processamento, converti todos os caracteres para minúsculas para unificar variações ortográficas e utilizei expressão regular (`[a-záàâãéêíóôõúçü]+`) para remover pontuações e numerais, preservando palavras com acentos típicos do português.
O vocabulário obtido no treino foi de 420 termos mantendo stopwords (Configs A e B) e de 401 termos após a remoção de stopwords (Configs C e D).
Defini uma lista de stopwords com palavras estritamente funcionais (como preposições, artigos e pronomes), porém com um cuidado crucial: a palavra "não" (bem como outros termos de negação) foi propositalmente preservada e nunca removida.
O termo "não" é um modificador de escopo que inverte completamente a polaridade semântica de uma oração; ao descartá-lo, uma frase negativa como "não gostei" seria reduzida a "gostei", induzindo o modelo de aprendizado supervisionado a classificar uma crítica severa como um elogio.

## 2. BoW vs. TF-IDF

Ao comparar as configurações por validação cruzada estratificada em 5 folds, as abordagens baseadas em Bag of Words superaram o TF-IDF: Config A e Config C atingiram F1-score médio idêntico de 0.8563 (acurácia média de 0.8564), enquanto o TF-IDF alcançou 0.8459 na Config B e 0.8354 na Config D.
Embora as médias estejam próximas dentro da margem de erro dos desvios padrão (entre 0.0661 e 0.0926), a representação BoW capturou melhor a frequência bruta de termos chave em um corpus de frases curtas (média de ~7.7 palavras por texto).
Optei pela Configuração A (BoW com stopwords mantidas) como vencedora por apresentar o maior F1 macro médio (0.8563) e reter termos funcionais essenciais de negação e ênfase no vocabulário.
Conceitualmente, o BoW quantifica apenas a contagem simples de ocorrência das palavras, ao passo que o TF-IDF pondera a frequência do termo penalizando palavras que ocorrem em muitos documentos (IDF), o que em textos muito concisos acabou atenuando o peso relativo de palavras críticas.

## 3. Avaliação no teste

No conjunto de teste independente (65 instâncias: 33 negativas e 32 positivas), a matriz de confusão revelou 25 verdadeiros negativos, 8 falsos positivos, 28 verdadeiros positivos e 4 falsos negativos.
Para a classe negativa, obtivemos precisão de 0.8621, recall de 0.7576 e F1 de 0.8065; para a positiva, precisão de 0.7778, recall de 0.8750 e F1 de 0.8235, resultando em acurácia geral de 0.8154 e macro F1 de 0.8150.
Em uma operação de e-commerce, classificar uma reclamação como elogio (falso positivo para positivo) é substancialmente mais grave, pois silencia um cliente insatisfeito, mascara falhas graves de atendimento/produto e impede ações rápidas de retenção pelo suporte.
A acurácia é uma métrica confiável e não enganosa neste cenário porque as classes do dataset são balanceadas (50% positivas e 50% negativas tanto no corpus geral quanto no split estratificado de teste), diferindo criticamente de problemas altamente desbalanceados como detecção de fraudes, onde prever sempre a classe majoritária mascara a ineficácia do modelo.

## 4. Validação cruzada vs. divisão única

A validação cruzada estratificada em 5 folds é muito mais confiável do que uma divisão simples, pois avalia o modelo em múltiplas partições do treino, atenuando a variância decorrente de divisões acidentais favoráveis ou desfavoráveis e fornecendo estimativas de desvio padrão.
A escolha da melhor configuração jamais deve ser realizada observando o conjunto de teste, pois usar o teste para tomada de decisão converte o conjunto de teste em parte indireta do treino, superajustando hiperparâmetros a uma amostra particular e eliminando sua capacidade de mensurar generalização futura.
O vazamento de dados (data leakage) ocorreria se o vetorizador (BoW/TF-IDF) fosse ajustado (`fit`) na totalidade dos dados antes da divisão ou dos folds, permitindo que a distribuição e vocabulário do teste contaminassem o aprendizado.
Esse problema foi rigorosamente prevenido com o uso da classe `Pipeline` do scikit-learn acoplada ao `cross_validate`, garantindo que o ajuste do vocabulário e a transformação ocorram exclusivamente dentro dos dados de treino de cada partição.

## 5. Análise de erros e palavras influentes

Avaliando o arquivo `erros.csv`, três casos ilustram bem as fragilidades do modelo: o texto "Panela de péssima qualidade, a comida gruda e queima" (negativo, predito positivo com prob. 0.6653) falhou devido à presença de termos neutros não vistos ou que receberam pesos espúrios; em "Ficou lindo na sala, combinou perfeitamente com os móveis" (positivo, predito negativo com prob. 0.4766), vocábulos como "lindo" não estavam presentes no vocabulário de treino; já em "Cheguei animado, mas me arrependi, tudo estava sujo" (negativo, predito positivo com prob. 0.6748), o peso de "animado" se sobrepôs à queixa.
Analisando as 10 palavras mais influentes, termos funcionais e neutros ganharam coeficientes significativos, como "não" (-2.0485), "dinheiro" (-0.6997), "toda" (-0.6795) e "mais" (-0.6295) para negativo, e "bastante" (+2.3408) e "bem" (+0.9118) para positivo.
A presença expressiva de palavras como "bastante", "toda" e "dinheiro" sem forte carga sentimental intrínseca revela que o classificador aprendeu correlações estatísticas superficiais e vícios frequentes do conjunto de dados sintético, em vez de uma compreensão semântica profunda de sentimentos.

## 6. Casos difíceis e limitações

No arquivo `casos_dificeis_predicoes.csv`, o modelo acertou negações diretas ("Não gostei nada do atendimento", previsto negativo), mas falhou em negações afirmativas ("Não tenho nada a reclamar, tudo perfeito", previsto negativo com prob. 0.4446) devido ao forte coeficiente negativo atribuído isoladamente ao termo "não" (-2.0485).
Em casos de ironia ("Que ótimo, o produto chegou quebrado", previsto positivo com prob. 0.8300) e frases mistas ("O produto é bom, mas a entrega foi péssima", previsto positivo com prob. 0.7920), o BoW colapsou por considerar apenas contagens desacopladas de palavras ("ótimo" e "bom"), ignorando sintaxe, ordem e conjunções adversativas.
BoW e TF-IDF sofrem dessa limitação estrutural pela premissa de bag of words (saco de palavras), que descarta completamente a ordem dos termos e a estrutura oracional; para atenuar isso, duas melhorias clássicas seriam a inclusão de n-gramas (bigramas e trigramas) para capturar composições como "não gostei" ou a integração de léxicos de sentimentos com regras de inversão de polaridade por janela contextual.
Quanto às limitações do dataset, ele é didático, muito reduzido (260 amostras) e gerado por templates sintéticos repetitivos, gerando um risco elevado em produção: o modelo falhará dramaticamente diante de gírias, erros de digitação, ambiguidades contextuais e sarcasmo comum em avaliações do mundo real.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: estruturação do ambiente local e execução dos scripts (código e análises elaborados de acordo com os critérios do projeto).
