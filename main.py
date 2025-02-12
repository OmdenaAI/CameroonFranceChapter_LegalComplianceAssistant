import os
import spacy
import random
import tempfile
import argparse
from spacy.tokens import DocBin
from datasets import load_dataset
from spacy.training.example import Example
from sklearn.model_selection import train_test_split


def get_train_data(dataset_name="conll2003"):
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    spacy_ner_tags = {
        1: "PERSON",
        2: "PERSON",
        3: "ORG",
        4: "ORG",
        5: "LOC",
        6: "LOC"
    }
    spans = []
    train_data = []

    for d in dataset["train"]:
        tokens = d["tokens"]
        current_index = 0
        spans = []

        for token_index, token in enumerate(tokens):
            spacy_tag = spacy_ner_tags.get(d["ner_tags"][token_index], None)
            if spacy_tag is not None:
                spans.append((current_index, current_index + len(token), spacy_tag))
            current_index += len(token) + 1
        sentence = " ".join(tokens)
        if sentence[-1] == ".":
            sentence = sentence[:-2] + "."

        train_data.append((sentence, {"spans": spans}))  

    return train_data


def get_train_doc_bins(docs, test_size=0.2, random_state=40):
    train_docs, val_docs = train_test_split(docs, test_size=test_size, random_state=random_state, shuffle=True)
    
    train_doc_bin = create_spacy_doc(train_docs)
    val_doc_bin = create_spacy_doc(val_docs)

    return train_doc_bin, val_doc_bin


def create_spacy_doc(data, language_model="en_core_web_trf"):
    doc_bin = DocBin()
    nlp = spacy.load(language_model)

    for text, annotations in data:
        doc = nlp.make_doc(text)
        spans = [doc.char_span(start, end, label, alignment_mode="expand") for start, end, label in annotations["spans"]]
        valid_spans = [s for s in spans if s is not None]
        doc.spans["sc"] = valid_spans
        doc_bin.add(doc)

    return doc_bin


def get_example(doc):
    example = Example.from_dict(doc, {"spans": {"sc": [
        (s.start_char, s.end_char, s.label_) 
        for s in doc.spans["sc"]]
    }})
    
    return example


def evaluate_model(model, data):
    examples = [get_example(doc) for doc in data]
    scorer = model.evaluate(examples)
    return scorer


def calc_val_loss(model, data):
    losses = {}

    for doc in data:
        example = get_example(doc)
        model.update([example], sgd=None, losses=losses, drop=0)

    return sum(losses.values())


def load_spacy_data(model, file_path):
    doc_bin = DocBin().from_disk(file_path)
    return list(doc_bin.get_docs(model.vocab))


def fine_tune_model(train_file_path, 
                    val_file_path, 
                    language_model="en_core_web_trf", 
                    epochs=3,
                    model_save_path=os.path.join(tempfile.gettempdir(), "fine_tuned_spancat_model_trf")):
    spacy.require_gpu()
    model = spacy.load(language_model)
    train_data = load_spacy_data(model, train_file_path)
    val_data = load_spacy_data(model, val_file_path)
    pipes_to_include = ["spancat", "transformer"]
    pipes_to_ignore = [p for p in model.pipe_names if p not in pipes_to_include]
    train_losses = []
    val_losses = []

    with model.disable_pipes(*pipes_to_ignore):
        opt = model.resume_training()
        opt.learn_rate = 0.00005

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} ...")
            losses = {}
            random.shuffle(train_data)

            for doc in train_data:
                example = get_example(doc)
                model.update([example], drop=0.15, losses=losses)

        val_loss = calc_val_loss(model, val_data)
        train_loss = sum(losses.values())
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1} - Training loss: {train_loss} | Validation loss: {val_loss}")
        scorer = evaluate_model(model, val_data)
        print(f"Validation score: {scorer}")

    model.to_disk(model_save_path)


def main(train_data_file_path, val_data_file_path):
    train_data = get_train_data()
    print(f"len(train_data): {len(train_data)}")
    train_doc_bin, val_doc_bin = get_train_doc_bins(train_data)
    train_doc_bin.to_disk(train_data_file_path)
    val_doc_bin.to_disk(val_data_file_path)
    fine_tune_model(train_data_file_path, val_data_file_path, epochs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-file-path", default="./data/conll2003_train.spacy", 
                        help="Path to the output file containing training data.")
    parser.add_argument("-v", "--val-file-path", default="./data/conll2003_val.spacy", 
                        help="Path to the output file containing validation data.")
    args = parser.parse_args()
    main(args.train_file_path, args.val_file_path)