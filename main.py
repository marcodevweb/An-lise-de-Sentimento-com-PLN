import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessamento import preprocessar_texto
from src.modelos import criar_pipeline, avaliar_configuracao_cv
from src.avaliacao import (
    gerar_relatorio_exploracao,
    salvar_matriz_confusao,
    salvar_relatorio_classificacao,
    extrair_erros,
    extrair_palavras_influentes,
    avaliar_casos_dificeis
)


def main():
    os.makedirs("resultados", exist_ok=True)

    df_avaliacoes = pd.read_csv("data/avaliacoes.csv")
    df_casos = pd.read_csv("data/casos_dificeis.csv")

    gerar_relatorio_exploracao(df_avaliacoes, "resultados/exploracao.txt")

    df_avaliacoes["texto_com_sw"] = df_avaliacoes["texto"].apply(lambda t: preprocessar_texto(t, remover_stopwords=False))
    df_avaliacoes["texto_sem_sw"] = df_avaliacoes["texto"].apply(lambda t: preprocessar_texto(t, remover_stopwords=True))

    idx_train, idx_test = train_test_split(
        df_avaliacoes.index,
        test_size=0.25,
        stratify=df_avaliacoes["sentimento"],
        random_state=42
    )

    y_train = df_avaliacoes.loc[idx_train, "sentimento"]
    y_test = df_avaliacoes.loc[idx_test, "sentimento"]

    configuracoes = [
        {"config": "A", "stopwords": "mantidas", "vetorizador": "BoW", "tipo_vet": "bow", "col_texto": "texto_com_sw"},
        {"config": "B", "stopwords": "mantidas", "vetorizador": "TF-IDF", "tipo_vet": "tfidf", "col_texto": "texto_com_sw"},
        {"config": "C", "stopwords": "removidas", "vetorizador": "BoW", "tipo_vet": "bow", "col_texto": "texto_sem_sw"},
        {"config": "D", "stopwords": "removidas", "vetorizador": "TF-IDF", "tipo_vet": "tfidf", "col_texto": "texto_sem_sw"},
    ]

    resultados_comparacao = []

    for cfg in configuracoes:
        X_tr = df_avaliacoes.loc[idx_train, cfg["col_texto"]]
        cv_res = avaliar_configuracao_cv(X_tr, y_train, vetorizador_tipo=cfg["tipo_vet"], random_state=42)
        
        pipe_temp = criar_pipeline(vetorizador_tipo=cfg["tipo_vet"], random_state=42)
        pipe_temp.named_steps["vetorizador"].fit(X_tr)
        tamanho_vocab = len(pipe_temp.named_steps["vetorizador"].vocabulary_)

        resultados_comparacao.append({
            "config": cfg["config"],
            "stopwords": cfg["stopwords"],
            "vetorizador": cfg["vetorizador"],
            "vocabulario": tamanho_vocab,
            "acc_cv_media": round(cv_res["acc_cv_media"], 4),
            "acc_cv_desvio": round(cv_res["acc_cv_desvio"], 4),
            "f1_cv_media": round(cv_res["f1_cv_media"], 4),
            "f1_cv_desvio": round(cv_res["f1_cv_desvio"], 4),
            "_tipo_vet": cfg["tipo_vet"],
            "_col_texto": cfg["col_texto"]
        })

    df_comp = pd.DataFrame(resultados_comparacao)
    colunas_salvar = ["config", "stopwords", "vetorizador", "vocabulario", "acc_cv_media", "acc_cv_desvio", "f1_cv_media", "f1_cv_desvio"]
    df_comp[colunas_salvar].to_csv("resultados/comparacao.csv", index=False)

    melhor_cfg_row = df_comp.sort_values(by=["f1_cv_media", "acc_cv_media"], ascending=[False, False]).iloc[0]

    X_train_final = df_avaliacoes.loc[idx_train, melhor_cfg_row["_col_texto"]]
    X_test_final = df_avaliacoes.loc[idx_test, melhor_cfg_row["_col_texto"]]

    modelo_final = criar_pipeline(vetorizador_tipo=melhor_cfg_row["_tipo_vet"], random_state=42)
    modelo_final.fit(X_train_final, y_train)

    y_pred_test = modelo_final.predict(X_test_final)
    y_prob_test = modelo_final.predict_proba(X_test_final)

    labels = ["negativo", "positivo"]
    salvar_matriz_confusao(y_test, y_pred_test, labels=labels, caminho_png="resultados/matriz_confusao.png", caminho_csv="resultados/matriz_confusao.csv")
    salvar_relatorio_classificacao(y_test, y_pred_test, caminho_saida="resultados/relatorio_classificacao.txt")

    df_test_full = df_avaliacoes.loc[idx_test].copy()
    extrair_erros(df_test_full, y_pred_test, y_prob_test, labels=labels, caminho_saida="resultados/erros.csv")
    extrair_palavras_influentes(modelo_final, caminho_saida="resultados/palavras_influentes.csv", top_n=10)

    remover_sw_casos = (melhor_cfg_row["stopwords"] == "removidas")
    df_casos["texto_limpo"] = df_casos["texto"].apply(lambda t: preprocessar_texto(t, remover_stopwords=remover_sw_casos))
    avaliar_casos_dificeis(modelo_final, df_casos, caminho_saida="resultados/casos_dificeis_predicoes.csv")


if __name__ == "__main__":
    main()
