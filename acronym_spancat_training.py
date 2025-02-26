import spacy
import consts
from datasets import load_dataset
from spacy.tokens import Doc, DocBin


#["B-long", "B-short", "I-long", "I-short", "O"]
LABELS = {
    0: "B-long",
    1: "B-short",
    2: "I-long",
    3: "I-short",
    4: "O"
}


def get_token_bounds_search_func(doc):
    cur_char_idx = 0

    def f(token):
        nonlocal cur_char_idx
        char_idx = doc.text.index(token, cur_char_idx)
        end_char_idx = char_idx + len(token)
        cur_char_idx = end_char_idx
        return char_idx, end_char_idx

    return f


def get_spans(doc, tokens, ents):
    get_token_bounds = get_token_bounds_search_func(doc)
    spans = [doc.char_span(*get_token_bounds(tokens[start]), label) 
                 for start, _, label in ents]
    
    return spans
          

def get_annots(nlp, data):
    doc_bin = DocBin()
    ner_pipe = nlp.get_pipe("ner")
    data_len = len(data)

    for index, row in enumerate(data):
        doc = Doc(nlp.vocab, words=row["tokens"])
        doc = ner_pipe(doc)
        ents = [(i, i + 1, LABELS[l]) for i, l in enumerate(row["labels"]) if l != 4]
        old_ents = [(e.start, e.end, e.label_) for e in doc.ents]
        spans_old_ents = get_spans(doc, row["tokens"], old_ents)
        spans = get_spans(doc, row["tokens"], ents)
        spans.extend(spans_old_ents)
        spans = [s for s in spans if s is not None]
        doc.spans["sc"] = spans
        doc_bin.add(doc)
        if index % 100 == 0:
            print(f"Processed {index/data_len:.2f}%")

    print("Processed 100%")

    return doc_bin


def main():
    nlp = spacy.load("en_core_web_trf")
    ds = load_dataset("amirveyseh/acronym_identification")

    train_annots = get_annots(nlp, ds["train"])
    val_annots = get_annots(nlp, ds["validation"])
    test_annots = get_annots(nlp, ds["test"])
    
    train_annots.to_disk(consts.ACRONYMS_TRAINING_FOLDER / "train.spacy")
    val_annots.to_disk(consts.ACRONYMS_TRAINING_FOLDER / "dev.spacy")
    test_annots.to_disk(consts.ACRONYMS_TRAINING_FOLDER / "test.spacy")


if __name__ == "__main__":
    main()