import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def gerar_relatorio_exploracao(df: pd.DataFrame, caminho_saida: str):
    total_linhas = len(df)
    distribuicao = df["sentimento"].value_counts()

    df_copy = df.copy()
    df_copy["num_palavras"] = df_copy["texto"].apply(lambda t: len(str(t).split()))
    tamanho_medio = df_copy.groupby("sentimento")["num_palavras"].mean()

    exemplos_pos = df[df["sentimento"] == "positivo"].head(3)
    exemplos_neg = df[df["sentimento"] == "negativo"].head(3)

    conteudo = []
    conteudo.append("=== EXPLORAÇÃO DOS DADOS ===")
    conteudo.append(f"Total de avaliações: {total_linhas}")
    conteudo.append("\nDistribuição das classes:")
    for classe, count in distribuicao.items():
        conteudo.append(f" - {classe}: {count} ({count/total_linhas*100:.1f}%)")

    conteudo.append("\nTamanho médio dos textos (em palavras):")
    for classe, media in tamanho_medio.items():
        conteudo.append(f" - {classe}: {media:.2f} palavras")

    conteudo.append("\n3 Exemplos de avaliações POSITIVAS:")
    for _, row in exemplos_pos.iterrows():
        conteudo.append(f" [ID {row['id']}] {row['texto']}")

    conteudo.append("\n3 Exemplos de avaliações NEGATIVAS:")
    for _, row in exemplos_neg.iterrows():
        conteudo.append(f" [ID {row['id']}] {row['texto']}")

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("\n".join(conteudo))


def salvar_matriz_confusao(y_real, y_pred, labels, caminho_png: str, caminho_csv: str = None):
    cm = confusion_matrix(y_real, y_pred, labels=labels)
    if caminho_csv:
        df_cm = pd.DataFrame(cm, index=[f"Real_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
        df_cm.to_csv(caminho_csv)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Matriz de Confusão - Conjunto de Teste")
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300)
    plt.close()


def salvar_relatorio_classificacao(y_real, y_pred, caminho_saida: str):
    rep = classification_report(y_real, y_pred, digits=4)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE CLASSIFICAÇÃO (TESTE FINAL) ===\n\n")
        f.write(rep)


def extrair_erros(df_teste: pd.DataFrame, y_pred, y_prob, labels, caminho_saida: str):
    df_erros = df_teste.copy()
    df_erros["previsto"] = y_pred

    pos_idx = list(labels).index("positivo") if "positivo" in labels else 1
    df_erros["probabilidade_positivo"] = y_prob[:, pos_idx]

    erros = df_erros[df_erros["sentimento"] != df_erros["previsto"]]
    erros = erros[["texto", "sentimento", "previsto", "probabilidade_positivo"]]
    erros.rename(columns={"sentimento": "real", "probabilidade_positivo": "probabilidade"}, inplace=True)
    erros.to_csv(caminho_saida, index=False, encoding="utf-8")
    return erros


def extrair_palavras_influentes(pipeline, caminho_saida: str, top_n: int = 10):
    vetorizador = pipeline.named_steps["vetorizador"]
    classificador = pipeline.named_steps["classificador"]

    feature_names = np.array(vetorizador.get_feature_names_out())
    coefs = classificador.coef_[0]

    indices_top_pos = np.argsort(coefs)[-top_n:][::-1]
    indices_top_neg = np.argsort(coefs)[:top_n]

    dados = []
    for idx in indices_top_pos:
        dados.append({
            "palavra": feature_names[idx],
            "coeficiente": coefs[idx],
            "sentimento_induzido": "positivo"
        })

    for idx in indices_top_neg:
        dados.append({
            "palavra": feature_names[idx],
            "coeficiente": coefs[idx],
            "sentimento_induzido": "negativo"
        })

    df_influentes = pd.DataFrame(dados)
    df_influentes.to_csv(caminho_saida, index=False, encoding="utf-8")
    return df_influentes


def avaliar_casos_dificeis(pipeline, df_casos: pd.DataFrame, caminho_saida: str):
    classes = list(pipeline.classes_)
    pos_idx = classes.index("positivo") if "positivo" in classes else 1

    probs = pipeline.predict_proba(df_casos["texto_limpo"])
    preds = pipeline.predict(df_casos["texto_limpo"])

    df_resultado = pd.DataFrame({
        "texto": df_casos["texto"],
        "tipo": df_casos["tipo"],
        "esperado": df_casos["sentimento_esperado"],
        "previsto": preds,
        "probabilidade_positivo": np.round(probs[:, pos_idx], 4)
    })

    df_resultado.to_csv(caminho_saida, index=False, encoding="utf-8")
    return df_resultado
