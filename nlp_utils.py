from sentence_transformers import SentenceTransformer


ENCODER_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  


def get_text_similarity(text1, text2):
    text1_embed = ENCODER_MODEL.encode(text1)
    text2_embed = ENCODER_MODEL.encode(text2)
    similarity = ENCODER_MODEL.similarity(text1_embed, text2_embed)

    return similarity