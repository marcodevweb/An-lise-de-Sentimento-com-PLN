import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def criar_pipeline(vetorizador_tipo: str = "bow", random_state: int = 42) -> Pipeline:
    if vetorizador_tipo.lower() == "bow":
        vetorizador = CountVectorizer()
    elif vetorizador_tipo.lower() == "tfidf":
        vetorizador = TfidfVectorizer()
    else:
        raise ValueError(f"Tipo de vetorizador desconhecido: {vetorizador_tipo}")

    classificador = LogisticRegression(max_iter=1000, random_state=random_state)

    pipeline = Pipeline([
        ("vetorizador", vetorizador),
        ("classificador", classificador)
    ])
    return pipeline


def avaliar_configuracao_cv(X_train: pd.Series, y_train: pd.Series, vetorizador_tipo: str, random_state: int = 42):
    pipeline = criar_pipeline(vetorizador_tipo=vetorizador_tipo, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False
    )

    return {
        "acc_cv_media": scores["test_accuracy"].mean(),
        "acc_cv_desvio": scores["test_accuracy"].std(),
        "f1_cv_media": scores["test_f1_macro"].mean(),
        "f1_cv_desvio": scores["test_f1_macro"].std()
    }
