import re

STOPWORDS_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as",
    "dos", "como", "mas", "ao", "das", "seu", "sua", "ou",
    "quando", "muito", "nos", "pelo", "pela", "ate", "isso", "ela", "ele"
}


def preprocessar_texto(texto: str, remover_stopwords: bool = False, stopwords: set = None) -> str:
    if not isinstance(texto, str):
        return ""

    texto_limpo = texto.lower()
    tokens = re.findall(r"[a-záàâãéêíóôõúçü]+", texto_limpo)

    if remover_stopwords:
        sw = stopwords if stopwords is not None else STOPWORDS_PT
        tokens = [token for token in tokens if token not in sw]

    return " ".join(tokens)
