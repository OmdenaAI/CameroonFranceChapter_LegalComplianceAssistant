import os
import io
import re
import sys
import fitz
import spacy
import string
import consts
import argparse
import tempfile
from pathlib import Path
from fontTools.ttLib import TTFont
from spacy.util import filter_spans
from spacy.language import Language
from spacy.pipeline import EntityRuler, Sentencizer
from sentence_transformers import SentenceTransformer


PATTERNS = {
    "email": re.compile(r"([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|" + \
                        r"\x22([^\x0d\x22\x5c\x80-\xff]|\x5c[\x00-\x7f])*\x22)(\x2e([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-" + \
                        r"\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|\x22([^\x0d\x22\x5c\x80-\xff]|\x5c[\x00-\x7f])*\x22))*\x40(" + \
                        r"[^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d\x7f-\xff]+|\x5b([^\x0d\x5b-\x5d\x80" + \
                        r"-\xff]|\x5c[\x00-\x7f])*\x5d)(\x2e([^\x00-\x20\x22\x28\x29\x2c\x2e\x3a-\x3c\x3e\x40\x5b-\x5d" + \
                        r"\x7f-\xff]+|\x5b([^\x0d\x5b-\x5d\x80-\xff]|\x5c[\x00-\x7f])*\x5d))*"),
    "ipv4": re.compile(r"(\b25[0-5]|\b2[0-4][0-9]|\b[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"),
    "ipv6": re.compile(r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|" + \
        r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|" + \
        r"([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|" + \
        r"([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|" + \
        r":((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|" + \
        r"::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|" + \
        r"(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|" + \
        r"1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))"),
    "phone": re.compile(r"([\+]\d+\s)?[(]?\d+[)]?[-\s\.]?\d+[-\s\.]?[0-9]{4,6}"),
    "ssn": re.compile(r"^(?!0{3})(?!6{3})[0-8]\d{2}-(?!0{2})\d{2}-(?!0{4})\d{4}$",),
    "medicare": re.compile(r"([a-z]{0,3})[-\s]?(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})[-\s]?([0-9a-z]{1,3})"),
    "vin": re.compile(r"\b[(A-H|J-N|P|R-Z|0-9)]{17}\b"),
    "url": re.compile(r"(((?:http[s]?:\/\/.)(?:www\.)?)|((?:ftp[s]?:\/\/.)))[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)")
}

COMMON_ACRONYMS = [
    "IANA",
    "Internet Assigned Numbers Authority",
    "HTML",
    "Hyper Text Markup Language",
    "MPLS",
    "Multi - Protocol Label Switching",
    "PSTN",
    "Public Switched Telecommunications Network",
    "Simple Mail Transfer Protocol",
    "SMTP"
]

ENTITY_DESC = {
    "ORG": ("Company", string),
    "LOC": ("Location", string),
    "PERSON": ("Person", string),
    "IPv4": ("IPv4 Address", int),
    "IPv6": ("IPv6 Address", int),
    "PHONE": ("Phone Number", int),
    "EMAIL": ("Email Address", int),
    "SSN": ("Social Security Number", int),
    "MEDICARE": ("Medicare Number", int),
    "VIN": ("Vehicle Identification Number", int),
    "URL": ("URL Address", int)
}

DEFAULT_PYMUPDF_FONTS = [
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Symbol", "ZapfDingbats"
]

DEFAULT_PYMUPDF_FONT = "Segoe UI" 


def find_matching_entities(doc, regex, label):
    matches = regex.finditer(doc.text)
    spans = [doc.char_span(match.start(), match.end(), label=label) for match in matches]
    spans = [s for s in spans if s is not None]
    spans = filter_spans(list(doc.ents) + spans)
    doc.ents = spans
    return doc


@Language.component("ipv4")
def ipv4_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ipv4"], "IPv4")
    return doc


@Language.component("ipv6")
def ipv6_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ipv6"], "IPv6")
    return doc


@Language.component("phone")
def phone_component(doc):
    doc = find_matching_entities(doc, PATTERNS["phone"], "PHONE")
    return doc


@Language.component("email")
def email_component(doc):
    doc = find_matching_entities(doc, PATTERNS["email"], "EMAIL")
    return doc
    

@Language.component("ssn")
def ssn_component(doc):
    doc = find_matching_entities(doc, PATTERNS["ssn"], "SSN")
    return doc


@Language.component("medicare")
def medicare_component(doc):
    doc = find_matching_entities(doc, PATTERNS["medicare"], "MEDICARE")
    return doc

@Language.component("vin")
def vin_component(doc):
    doc = find_matching_entities(doc, PATTERNS["vin"], "VIN")
    return doc

@Language.component("url")
def url_component(doc):
    doc = find_matching_entities(doc, PATTERNS["url"], "URL")
    return doc


@Language.component("custom_sentencizer")
def get_sentencizer(doc):
    sentencizer = Sentencizer(punct_chars=[r"\n"])
    return sentencizer(doc)


def get_spans_similarity(span1, span2, text_encoder):
    span1_embedding = text_encoder.encode(span1)
    span2_embedding = text_encoder.encode(span2)
    similarity = text_encoder.similarity(span1_embedding, span2_embedding)

    return similarity


def check_common_acronyms(span_text, common_acronyms, text_encoder, similarity_score_threshold=0.95):
    max_similarity = 0
    optim_acronym = None
    for acronym in common_acronyms:
        span_similarity_score = get_spans_similarity(acronym, span_text, text_encoder)
        if (span_similarity_score > max_similarity) and \
                    (span_similarity_score > similarity_score_threshold):
                    optim_acronym = acronym
                    max_similarity = span_similarity_score

    return optim_acronym


def get_most_similar_text(texts_to_compare, texts, text_encoder, similarity_score_threshold=0.95):
    most_similar_text = None
    
    if (texts_to_compare is not None) and (texts is not None):
        curr_max_similarity = 0
        for text1 in texts_to_compare:
            for text2 in texts:
                if isinstance(text2, tuple):
                    text_similarity_score = max(
                        get_spans_similarity(text2[0], text1, text_encoder),
                        get_spans_similarity(text2[1], text1, text_encoder))
                else:
                    text_similarity_score = get_spans_similarity(text2, text1, text_encoder)       
                if (text_similarity_score > curr_max_similarity) and \
                    (text_similarity_score > similarity_score_threshold):
                    most_similar_text = text2
                    curr_max_similarity = text_similarity_score

    return most_similar_text


def get_ent_replacement(ent, suffix_type, entities_count=0):
    if suffix_type is string:
        suffix = int(round(entities_count / len(string.ascii_uppercase))) * "A" + \
            string.ascii_uppercase[entities_count % len(string.ascii_uppercase)]
        return (
            f"\"{ENTITY_DESC[ent.label_][0]} {suffix}\"", 
            ent.start_char,
            ent.end_char,
            ent.sent.start_char,
            ent.sent.end_char
        )
    elif suffix_type is int:
        return (
            f"\"{ENTITY_DESC[ent.label_][0]} {entities_count+1}\"", 
            ent.start_char,
            ent.end_char,
            ent.sent.start_char,
            ent.sent.end_char
        )
    else:
        raise Exception(f"Unsupported suffix type '{suffix_type}")
             

def get_closest_ent_name(texts, ent_cat_subs, relations, text_encoder):
    relation = get_most_similar_text(texts, relations, text_encoder)
    closest_ent_name = None
    if relation:
        closest_ent_name = relation[0] if relation[0] in ent_cat_subs \
            else relation[1] if relation[1] in ent_cat_subs \
            else None
        if not closest_ent_name:
            closest_ent_name = get_most_similar_text(texts, ent_cat_subs, text_encoder)
    return closest_ent_name


def update_subs(subs, ents, relations, common_acronyms, text_encoder, paragraph_index):
    for ent in ents:
        if ent.label_ not in ENTITY_DESC:
            continue
        if ent.label_ == "ORG":
            acronym = check_common_acronyms(ent.text, common_acronyms, text_encoder)
            if acronym:
                continue
        ent_subs = subs.get(ent.label_, {})
        suffix_type = ENTITY_DESC[ent.label_][1]
            
        if ent.label_ == "ORG":
            texts_to_compare = [ent.text]
            relation = get_most_similar_text([ent.text], relations, text_encoder)
            if relation:
                texts_to_compare.extend([i for i in relation])
            closest_key_name = get_closest_ent_name(texts_to_compare, ent_subs, relations + list(ent_subs.keys()), text_encoder)

            if closest_key_name:
                ent_replacements = ent_subs[closest_key_name]
                ent_replacements.append((
                    # Pick the first replacement item for the entity 
                    # name and select its replacement string
                    ent_subs[closest_key_name][0][0],
                    ent.start_char,
                    ent.end_char,
                    ent.sent.start_char,
                    ent.sent.end_char,
                    paragraph_index
                ))
                continue

            ent_subs[ent.text] = [(*get_ent_replacement(ent, suffix_type, len(subs.get(ent.label_, {}).keys())), paragraph_index)]
        else:
            if ent.text in ent_subs:
                replacements = ent_subs[ent.text]
                replacements.append((
                    replacements[0][0],
                    ent.start_char,
                    ent.end_char,
                    ent.sent.start_char,
                    ent.sent.end_char,
                    paragraph_index
                ))
            else:
                replacements = [(*get_ent_replacement(ent, suffix_type, len(subs.get(ent.label_, {}).keys())), paragraph_index)]
                ent_subs[ent.text] = replacements
            
        subs[ent.label_] = ent_subs

    return subs


def get_lines_subs(subs):
    lines_subs = {}

    for ent_label in subs:
        for old_val, occurences_list in subs[ent_label].items():
            for (new_val, sent_text) in occurences_list:
                lines = lines_subs.get(sent_text, [])
                new_item = (ent_label, old_val, new_val)
                if new_item not in lines:
                    lines.append(new_item)
                    lines_subs[sent_text] = lines

    return lines_subs


def is_acronym(short_form, long_form_entities):
    long_acronym = ''.join(ent.text[0].upper() for ent in long_form_entities)
    long_acronym = re.sub(r"[^\w]", "", long_acronym)
    return short_form.upper() == long_acronym


def find_short_long_relations(acronyms_ents):
    if len(acronyms_ents) == 0:
        return
    long_form = []
    long_form_range = (0, 0)
    relations = []
    
    for ent in acronyms_ents:
        tag, _, start, end = ent.label_, ent.text, ent.start, ent.end
        if (tag.startswith("B-long") or tag.startswith("I-long")) and \
            start - long_form_range[1] > 2:
            long_form = [(ent)]
            long_form_range = (start, end)
        elif (tag.startswith("I-long")) or \
             ((tag.startswith("B-long") and \
             (start - long_form_range[1]) <=2)):
            long_form.append(ent)
            long_form_range = (long_form_range[0], end)
        elif tag.startswith("B-short"):
           if long_form and (start - long_form_range[1]) <= 2: 
                if is_acronym(ent.text, long_form) or ((start - long_form_range[1] <= 2) and \
                                                    (long_form[-1].sent == ent.sent)):
                    relations.append((" ".join([e.text for e in long_form]), ent.text))
                    long_form_range = (0, 0)

    return relations


def get_font_metadata(font_path):
    font = TTFont(font_path)
    metadata = {}

    for record in font["name"].names:
        name = record.toUnicode()
        metadata[record.nameID] = name

    return {
        "family": metadata.get(1, "Unknown"),
        "sub_family": metadata.get(2, "Unknown"),
        "full_name": metadata.get(4, "Unknown"),
        "version": metadata.get(5, "Unknown")
    }


def get_font_path(font_name):
    """Find the system path of a given font by name."""
    if sys.platform == "win32":
        font_dirs = [Path("C:/Windows/Fonts")]
    elif sys.platform == "darwin":  # macOS
        font_dirs = [Path("~/Library/Fonts").expanduser(), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts")]
    else:  # Linux
        font_dirs = [Path("~/.fonts").expanduser(), Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]

    for font_dir in font_dirs:
        if font_dir.exists():
            for font_path in font_dir.rglob("*.ttf"):
                font_metadata = get_font_metadata(font_path)
                if font_name == f"{font_metadata['family']}" or \
                   font_name == f"{font_metadata["family"]}-{font_metadata["sub_family"]}":
                    return str(font_path)

    return None


def load_font(font_name):
    font_path = get_font_path(font_name)
    if font_path:
        with open(font_path, "rb") as fh:
            return io.BytesIO(fh.read())

    return None


def get_iou(bbox1, bbox2):
    x0, y0, x1, y1 = bbox1
    xx0, yy0, xx1, yy1 = bbox2
    max_x0 = max(x0, xx0)
    max_y0 = max(y0, yy0)
    min_x1 = min(x1, xx1)
    min_y1 = min(y1, yy1)
    intersection_area = (min_x1 - max_x0) * (min_y1 - max_y0)
    union_area = (x1 - x0) * (y1 - y0) + (xx1 - xx0) * (yy1 - yy0) - intersection_area
    
    return intersection_area / union_area


def is_span_a_link(line_subs, links, iou_thres=0.5):
    overlapping_links = []
    for line_sub in line_subs:
        for link in links:
            iou = get_iou(line_sub["spans_bbox"], link["from"])
            if iou > iou_thres:
                overlapping_links.append(link)
    return overlapping_links


def get_updated_text(line_subs):
    updated_text = line_subs[0]["text"]

    for sub in line_subs:
        new_val = sub["new_val"]
        old_val = sub["old_val"]
        if len(new_val) < len(old_val):
            if sub["ent_start_char"] == 0:
                new_val = new_val.ljust(len(old_val))
            elif sub["ent_end_char"] == (sub["end_char"]-sub["start_char"]):
                new_val = new_val.rjust(len(old_val))
            else:
                new_val = new_val.center(len(old_val))
        elif len(new_val) > len(old_val):
            new_val_parts = new_val.split(" ")
            stripped_chars_count = len(new_val)-len(old_val)
            new_val = " ".join(new_val_parts[:-1])[:-(stripped_chars_count+1)] + ". " + new_val_parts[-1] 
        updated_text = updated_text[:sub["ent_start_char"]] + \
            new_val + updated_text[sub["ent_end_char"]:]
        
    return updated_text
        

def replace_text(pdf, page_num, line_subs, page_hyperlinks):
    if line_subs:
        updated_text = get_updated_text(line_subs)
        x, y = line_subs[0]["origin"]
        xb0, yb0, xb1, yb1 = line_subs[0]["spans_bbox"]#line_bbox#line_subs["bbox"]
        rect = fitz.Rect(xb0, yb0, xb1, yb1)
        page = pdf[page_num]#pdf[line_subs["page_number"]]
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        text_color = line_subs[0]["color"] if isinstance(line_subs[0]["color"], tuple) \
            else [round(c/255.0, 2) for c in fitz.sRGB_to_rgb(line_subs[0]["color"])]
        kwargs = dict(fontsize=line_subs[0]["size"], 
                      color=text_color, 
                      fill_opacity=line_subs[0]["alpha"])
        
        if line_subs[0]["font"] in DEFAULT_PYMUPDF_FONTS:
            kwargs["fontname"] = line_subs[0]["font"]
        else:
            font_path = get_font_path(line_subs[0]["font"])
            if font_path:
                kwargs["fontfile"] = font_path
            else:
                kwargs["fontfile"] = get_font_path(DEFAULT_PYMUPDF_FONT)

        page.insert_text((x, y), updated_text, **kwargs)
        if links := is_span_a_link(line_subs, page_hyperlinks):
            for link in links:
                page.delete_link(link)


def substitute_page_entities(pdf, lines_subs, min_distance=3):
    for (page_num, _), occurences in lines_subs.items():
        replacements = []
        orgs_to_replace = []
        page_hyperlinks = pdf[page_num].get_links()
    
        for line_subs in occurences:
            if line_subs["label_type"] == "ORG":
                if orgs_to_replace:
                    last_org = orgs_to_replace[-1]
                    if line_subs["text"] == last_org["text"]:
                        # It's assumed that the input text into the spaCy's model is at least 
                        # normalized to lower case characters.
                        lower_case_sent = last_org["text"].lower()
                        if (line_subs["ent_start_char"] - last_org["ent_end_char"] <= min_distance) and \
                            (last_org["new_val"] == line_subs["new_val"]):
                            # The new ORG will not be added, instead the previous ORG's ending character index 
                            # will be expanded so that it includes the new ORG. 
                            new_org_end_char = line_subs["ent_end_char"]
                            if lower_case_sent[new_org_end_char + 1] == ")":
                                new_org_end_char += 1
                            orgs_to_replace[-1]["ent_end_char"] = new_org_end_char
                            continue
                orgs_to_replace.append(line_subs)
            else:
                #line_text_orig = re.sub(r"\s.\s?$", "", line)
                #old_val_orig = re.sub(r"\s.\s?$", "", old_val)
                replacements.append(line_subs)

        for org_line_sub in orgs_to_replace:
            replacements.append(line_subs)
            #sent_text_orig = re.sub(r"\s.\s?$", "", sent_text)
            #old_org_name_orig = re.sub(r"\s.\s?$", "", old_org_name)

        replace_text(pdf, page_num, replacements, page_hyperlinks)


def run_nlp(paragraphs):
    nlp = spacy.load("en_core_web_trf")
    nlp_acronyms = spacy.load(consts.ACRONYMS_MODEL_DIR)
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  

    components = ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]}]
    ruler.add_patterns(patterns)

    #nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe('sentencizer')
    nlp_acronyms.add_pipe('sentencizer')
    for c in components:
        nlp.add_pipe(c, last=True)

    subs = {} 
    for idx, paragraph in enumerate(paragraphs):
        doc = nlp(get_normalized_text(paragraph["text"]))
        acronyms_doc = nlp_acronyms(paragraph["text"])

        relations = find_short_long_relations(acronyms_doc.ents)

        for ent in doc.ents:
            print(ent.text, ent.label_)

        update_subs(subs, doc.ents, relations, COMMON_ACRONYMS, encoder_model, idx)

    lines_subs = map_subs_to_lines(subs, paragraphs)

    return lines_subs


def get_pdf_text(pdf):
    raw_text = ""

    for page in pdf:
        raw_text += page.get_text("text") + "\n"

    return raw_text


def get_pdf_spans(pdf):
    pdf_spans = {}
    
    for page in pdf:    
        spans = get_pdf_page_spans(page)
        for span, occurences in spans.items():
            if span in pdf_spans:
                pdf_spans[span].extend(occurences)
            else:
                pdf_spans[span] = occurences

    return pdf_spans


def get_pdf_span_boundaries(span):
    a = span["ascender"]
    d = span["descender"]
    r = fitz.Rect(span["bbox"])
    o = fitz.Point(span["origin"])
    r.y1 = o.y - span["size"] * d / (a - d)
    r.y0 = r.y1 - span["size"]

    return r


def get_pdf_page_spans(page):
    pdf_spans = {}
    blocks = page.get_text("dict")["blocks"]

    for block_idx, block in enumerate(blocks):
        is_text_block = block["type"] == 0
        if is_text_block:
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                spans = []
                for span in line["spans"]:
                    span_text = span["text"]
                    if span_text.strip() != "":
                        spans.append({
                            "page_number": page.number,
                            "block_index": block_idx,
                            "line_index": line_idx,
                            "size": span["size"],
                            "font": span["font"],#"helv", # Temporarly use the 'helv' font instead of the span["font"],
                            "color": span["color"],
                            "alpha": span["alpha"],
                            "origin": span["origin"],
                            "bbox": span["bbox"],
                            "span_bbox": get_pdf_span_boundaries(span),
                            # Use original text instead of the whitespaces the stripped text 
                            # will be used to match tet of sentences returned by the spaCy model.
                            "text": span_text
                        })
                        line_text += span_text
                if line_text != "":
                    line_span = spans[0].copy()
                    line_span["bbox"] = line["bbox"]
                    spans_max_y = max([s["span_bbox"][-1] for s in spans])
                    line_span["spans_bbox"] = ((spans[0]["span_bbox"][0], spans[0]["span_bbox"][1], spans[-1]["span_bbox"][-2], spans_max_y))
                    line_span["text"] = line_text
                    key_name = line_text
                    occurences = pdf_spans.get(key_name, [])
                    occurences.extend([line_span])
                    pdf_spans[key_name] = occurences

    return pdf_spans


def get_normalized_text(text):
    return text.lower()


def extract_pdf_text(pdf, line_gap_threshold=5):
    all_paragraphs = []
    sorted_lines = []

    pdf_spans = get_pdf_spans(pdf)
    for line, infos in pdf_spans.items():
        for line_info in infos:
            temp_line_info = line_info.copy()
            sorted_lines.append(temp_line_info)

    sorted_lines = list(sorted(sorted_lines, key=lambda l: (l["page_number"], l["spans_bbox"][-1])))

    paragraphs = []
    current_paragraph = []
    last_y = None
    char_offset = 0
    
    for line in sorted_lines:
        x0, y0, x1, y1 = line["spans_bbox"]

        if last_y is not None and (y0 - last_y) > line_gap_threshold:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = []
            char_offset = 0

        temp_line = line.copy()
        temp_line["spans_bbox"] = (x0, y0, x1, y1)
        temp_line["start_char"] = char_offset
        temp_line["end_char"] = char_offset + len(line["text"])
        current_paragraph.append(temp_line)
        # Each line will be separated by a whitespace character.
        char_offset += len(line["text"]) + 1
        last_y = y1

    if current_paragraph:
        paragraphs.append(current_paragraph)

    for paragraph in paragraphs:
        text = " ".join([line["text"] for line in paragraph])
        all_paragraphs.append(dict(text=text, 
                                   lines=paragraph, 
                                   page_number=paragraph[0]["page_number"]))

    return all_paragraphs


def map_subs_to_lines(subs, paragraph):
    lines_subs = {}

    for label_type in subs:
        for old_val, replacements in subs[label_type].items():
            for (new_val, ent_start, ent_end, _, _, paragraph_index) in replacements:
                for line in paragraph[paragraph_index]["lines"]:
                    if (ent_start >= line["start_char"]) and \
                        (ent_end <= line["end_char"]):
                        line_subs = lines_subs.get((line["page_number"], line["bbox"]), [])
                        temp_line = line.copy()
                        temp_line["label_type"] = label_type
                        temp_line["ent_start_char"] = ent_start - line["start_char"]
                        temp_line["ent_end_char"] = ent_end - line["start_char"]
                        temp_line["old_val"] = old_val
                        temp_line["new_val"] = new_val
                        line_subs.append(temp_line)
                        lines_subs[(line["page_number"], line["bbox"])] = line_subs

    return lines_subs


def main(input_file, output_file=os.path.join(tempfile.gettempdir(), "test.pdf")):
    input_file_path = Path(input_file)
    pdf = fitz.open(input_file_path.absolute().as_posix())
    paragraphs = extract_pdf_text(pdf)
    lines_subs = run_nlp(paragraphs)
    substitute_page_entities(pdf, lines_subs)
    pdf.subset_fonts()
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.ez_save(output_file_path.absolute().as_posix())
    pdf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    
    main(args.input, args.output)