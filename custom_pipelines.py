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
from PIL import Image, ImageDraw
from metrics_utils import get_iou
from fontTools.ttLib import TTFont
from spacy.util import filter_spans
from spacy.language import Language
from multiprocessing import Pool, Queue
from spacy.pipeline import EntityRuler, Sentencizer
from sentence_transformers import SentenceTransformer
from analyze_images import analyze_print, analyze_handwritting

fitz.TOOLS.set_small_glyph_heights(True)


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

DEFAULT_TEXT_MEASUREMENT_FONT = "helv"
DEFAULT_PYMUPDF_FONT = {
    "normal": "Segoe UI",
    "italic": "Segoe UI Italic",
    "bold": "Segoe UI Bold"
}
DEFAULT_FONT_SIZE = 11


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
            temp_relations = (relations if relations else []) + list(ent_subs.keys())
            closest_key_name = get_closest_ent_name(texts_to_compare, ent_subs, temp_relations, text_encoder)

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


def get_text_length(text, font_name=DEFAULT_PYMUPDF_FONT, font_size=DEFAULT_FONT_SIZE):
    if font_name not in DEFAULT_PYMUPDF_FONTS or \
        font_name in DEFAULT_PYMUPDF_FONT.values() or \
            not get_font_path(font_name):
        font_name = DEFAULT_TEXT_MEASUREMENT_FONT
        
    return sum([fitz.get_text_length(c, fontname=font_name, fontsize=font_size) 
                for c in text])


def get_spans_text_length(spans):
    text_len = 0

    for span in spans:
        text_len += get_text_length(span["text"], span["font"], span["size"])

    return text_len


def get_redacted_text(line_subs, redact_char="x"):
    redacted_text = line_subs[0]["text"]
    offset = 0
    sorted_line_subs = list(sorted(line_subs, key=lambda i: i["ent_start_char"], reverse=True))

    for sub in sorted_line_subs:
        font = sub["spans"][0]["font"]
        size = sub["spans"][0]["size"]
        
        new_val = len(sub["old_val"]) * redact_char
        while get_text_length(new_val, font_name=font, font_size=size) >= \
            get_text_length(sub["old_val"], font_name=font, font_size=size):
            new_val = new_val[:-1]
        redacted_text = redacted_text[:sub["ent_start_char"] + offset] + \
            new_val + redacted_text[sub["ent_end_char"] + offset:]

    return redacted_text


def add_redact_annots(page, line_subs, use_span_bg=True, fill_color=(0, 0, 0)):
    annots = []

    if line_subs:
        for line_sub in line_subs:
            rects = page.search_for(line_sub["old_val"], clip=line_sub["spans_bbox"], quads=False, flags=0)
            for r in rects:
                draw_redact_rect = [r[0], line_sub["spans_bbox"][1] - 2, r[2], line_sub["spans_bbox"][3]]
                center_y = r[1] + (r[3] - r[1]) / 2
                redact_rect = [r[0], center_y - 0.1, r[2], center_y + 0.1]
                annots.append((draw_redact_rect, line_sub))
                redact_annot_color = line_sub["sub_spans"][0]["fill_color"] if use_span_bg else fill_color
                page.add_redact_annot(redact_rect, fill=redact_annot_color)
    
    return annots


def fill_bg(page, line):
    for span in line["spans"]:
        page.draw_rect(span["span_bbox"], color=span["fill_color"], fill=span["fill_color"])


def replace_text(page, line, text):
    for span in line["spans"]:
        text_color = span["color"] if isinstance(span["color"], tuple) \
            else [round(c/255.0, 2) for c in fitz.sRGB_to_rgb(span["color"])]
        kwargs = dict(fontsize=span["size"], 
                    color=text_color, 
                    fill_opacity=span["alpha"])
        
        font_name = span["font"]
        if span["font"] in DEFAULT_PYMUPDF_FONTS:
            kwargs["fontname"] = font_name
        else:
            font_path = get_font_path(font_name)
            if font_path:
                kwargs["fontfile"] = font_path
            else:
                font_name = DEFAULT_PYMUPDF_FONT["bold"] if "bold" in font_name \
                    else DEFAULT_PYMUPDF_FONT["italic"] if "italic" in font_name \
                    else DEFAULT_PYMUPDF_FONT["normal"]
                kwargs["fontfile"] = get_font_path(font_name)
        span_length = span["span_bbox"][2] - span["span_bbox"][0]
        length_ratio = get_text_length(text[span["text_start"]:span["text_end"]], 
                                        font_name=font_name, 
                                        font_size=span["size"]) / span_length
        if length_ratio == 0.0:
            length_ratio = 1.0
        kwargs["fontsize"] = span["size"] / length_ratio    
        x, y = span["origin"]
        page.insert_text((x, y), text[span["text_start"]:span["text_end"]], **kwargs)


def is_neighbouring_line(line, lines):
    for line_idx in lines:
        if abs(line["paragraph_line_index"] - line_idx) == 1:
            return True
    return False


def get_redacted_paragraphs(replacements):
    redacted_paragraphs = {}

    for redact_rect, replacement in replacements:
        key = (replacement["page_number"], replacement["paragraph_index"])
        parag_replacements = redacted_paragraphs.get(key, [])
        parag_replacements.append((redact_rect, replacement))
        redacted_paragraphs[key] = parag_replacements

    return redacted_paragraphs


def draw_paragraph_annots(page, line, redacted_lines, parag_replacements, redact_rect_color=(0, 0, 0)):
    if line["paragraph_line_index"] in redacted_lines:
        line_subs = [r for r in parag_replacements 
                        if r[1]["paragraph_line_index"] == line["paragraph_line_index"]]
        for redact_rect, _ in line_subs:
            page.draw_rect(redact_rect,
                                    color=redact_rect_color,
                                    fill=redact_rect_color)
            

def reconstruct_deleted_text(pdf, replacements, paragraphs):
    redacted_paragraphs = get_redacted_paragraphs(replacements)

    for (page_num, paragraph_index), parag_replacements in redacted_paragraphs.items():
        redacted_lines = [r[1]["paragraph_line_index"] for r in parag_replacements]

        for line in paragraphs[paragraph_index]["lines"]:
            if line["paragraph_line_index"] in redacted_lines or \
                is_neighbouring_line(line, redacted_lines):
                if line["paragraph_line_index"] in redacted_lines:
                    text = get_redacted_text([r[1] for r in parag_replacements 
                                              if r[1]["paragraph_line_index"] == line["paragraph_line_index"]])
                else:
                    text = line["text"]

                fill_bg(pdf[page_num], line)                
                replace_text(pdf[page_num], line, text)
                draw_paragraph_annots(pdf[page_num], line, redacted_lines, parag_replacements)
                

def substitute_page_entities(pdf, lines_subs, min_distance=3):
    all_replacements = []

    for (page_num, _), occurences in lines_subs.items():
        replacements = []
        orgs_to_replace = []
    
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
            replacements.append(org_line_sub)
            #sent_text_orig = re.sub(r"\s.\s?$", "", sent_text)
            #old_org_name_orig = re.sub(r"\s.\s?$", "", old_org_name)

        page = pdf[page_num]

        if replacements:
            annots = add_redact_annots(page, replacements)
            if annots:
                for (draw_redact_rect, _) in annots:
                    page.draw_rect(draw_redact_rect, color=(0, 0, 0), fill=(0, 0, 0))
                all_replacements.extend(annots)

    return all_replacements


def load_nlp_model(model_name="en_core_web_trf"):
    nlp = spacy.load(model_name)
    components = ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]}]
    ruler.add_patterns(patterns)
    #nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe('sentencizer')
    for c in components:
        nlp.add_pipe(c, last=True)

    return nlp


def load_nlp_acronyms_model():
    nlp_acronyms = spacy.load(consts.ACRONYMS_MODEL_DIR)
    nlp_acronyms.add_pipe('sentencizer')

    return nlp_acronyms


def run_nlp(paragraphs):
    nlp = load_nlp_model()
    nlp_acronyms = load_nlp_acronyms_model()
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  

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


def get_fill_color(span_bbox, rectangles, margin=1.5, iou_threshold=0.35):
    expanded_span_bbox = tuple(i + j for i, j in zip(span_bbox, (-margin, -margin, margin, margin)))
    intersect_rects = list(filter(lambda i: fitz.Rect(i["rect"]).intersects(expanded_span_bbox),
                                  rectangles))
    fill_color = (1, 1, 1)

    if intersect_rects:
        iou = get_iou(intersect_rects[0]["rect"], span_bbox)
        best_iou = iou if iou > iou_threshold else 0
        fill_color = intersect_rects[0]["fill"] if iou > 0 else fill_color 

        for rect in intersect_rects[1:]:
            iou = get_iou(rect["rect"], span_bbox)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                fill_color = rect["fill"]
        
    return fill_color


def get_pdf_page_spans(page):
    pdf_spans = {}
    blocks = page.get_text("dict")["blocks"]
    drawings = list(sorted(page.get_drawings(), key=lambda i: i["rect"][0]))
    rectangles = [r for r in drawings if r["items"][0][0] == "re"]

    for block_idx, block in enumerate(blocks):
        is_text_block = block["type"] == 0
        if is_text_block:
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                spans = []
                for span in line["spans"]:
                    span_text = span["text"]
                    if span_text.strip() != "":
                        span_bbox = get_pdf_span_boundaries(span)
                        fill_color = get_fill_color(span_bbox, rectangles)

                        if fill_color != (1.0, 1.0, 1.0):
                            print(f"Text: {span['text']}, fill color: {fill_color}")

                        spans.append({
                            "size": span["size"],
                            "font": span["font"],#"helv", # Temporarly use the 'helv' font instead of the span["font"],
                            "color": span["color"],
                            "fill_color": fill_color,
                            "alpha": span["alpha"],
                            "origin": span["origin"],
                            "bbox": span["bbox"],
                            "span_bbox": span_bbox,#get_pdf_span_boundaries(span),
                            # Use original text instead of the whitespaces the stripped text 
                            # will be used to match tet of sentences returned by the spaCy model.
                            "text": span_text,
                            "text_start": len(line_text),
                            "text_end": len(line_text) + len(span_text)
                        })
                        line_text += span_text
                if line_text != "":
                    line_info = {"page_number": page.number}
                    line_info["block_index"] = block_idx
                    line_info["line_index"] = line_idx
                    line_info["line_bbox"] = line["bbox"]
                    spans_max_y = max([s["span_bbox"][-1] for s in spans])
                    line_info["spans_bbox"] = ((spans[0]["span_bbox"][0], spans[0]["span_bbox"][1], spans[-1]["span_bbox"][-2], spans_max_y))
                    line_info["text"] = line_text
                    line_info["spans"] = spans
                    occurences = pdf_spans.get(line_text, [])
                    occurences.extend([line_info])
                    pdf_spans[line_text] = occurences

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
    paragraph_line_index = 0

    for line in sorted_lines:
        x0, y0, x1, y1 = line["spans_bbox"]

        if last_y is not None and (y0 - last_y) > line_gap_threshold:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = []
            char_offset = 0
            paragraph_line_index = 0

        temp_line = line.copy()
        temp_line["spans_bbox"] = (x0, y0, x1, y1)
        temp_line["start_char"] = char_offset
        temp_line["end_char"] = char_offset + len(line["text"])
        temp_line["paragraph_line_index"] = paragraph_line_index
        current_paragraph.append(temp_line)
        # Each line will be separated by a whitespace character.
        char_offset += len(line["text"]) + 1
        last_y = y1
        paragraph_line_index += 1

    if current_paragraph:
        paragraphs.append(current_paragraph)

    for paragraph in paragraphs:
        text = " ".join([line["text"] for line in paragraph])
        all_paragraphs.append(dict(text=text, 
                                   lines=paragraph, 
                                   page_number=paragraph[0]["page_number"]))

    return all_paragraphs


def map_subs_to_lines(subs, paragraphs):
    lines_subs = {}

    for label_type in subs:
        for old_val, replacements in subs[label_type].items():
            for (new_val, ent_start, ent_end, _, _, paragraph_index) in replacements:
                for line in paragraphs[paragraph_index]["lines"]:
                    if (ent_start >= line["start_char"]) and \
                        (ent_end <= line["end_char"]):
                        line_subs = lines_subs.get((line["page_number"], line["line_bbox"]), [])
                        temp_line = line.copy()
                        temp_line["label_type"] = label_type
                        temp_line["ent_start_char"] = ent_start - line["start_char"]
                        temp_line["ent_end_char"] = ent_end - line["start_char"]
                        temp_line["old_val"] = old_val
                        temp_line["new_val"] = new_val
                        temp_line["paragraph_index"] = paragraph_index
                        temp_line["sub_spans"] = []
                        for span in temp_line["spans"]:
                            if (temp_line["ent_start_char"] >= span["text_start"] and 
                                temp_line["ent_start_char"] < span["text_end"]) or \
                               (temp_line["ent_end_char"] > span["text_start"] and 
                                temp_line["ent_end_char"] <= span["text_end"]):
                                temp_line["sub_spans"].append(span)
                        line_subs.append(temp_line)
                        lines_subs[(line["page_number"], line["line_bbox"])] = line_subs

    return lines_subs


def extract_pdf_images(pdf):
    images = []

    for page in pdf:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            images.append({
                "xref": xref,
                "page_num": page.number,
                "ext": base_image["ext"], 
                "data": Image.open(io.BytesIO(image_bytes))
            })

    return images


def extract_pdf_drawings(pdf):
    drawings = {}

    for page in pdf:
        drawings[page.number] = page.get_drawings()

    return drawings


def save_images(images, dest_dir=tempfile.gettempdir()):
    for img in images:
        img_name = f"{img['page_num']}_{img['xref']}.{img['ext']}"
        img_path = os.path.join(dest_dir, img_name)
        img["data"].save(img_path)
        yield img_path
    

def save_drawings(drawings, page_width, page_height, dest_dir=tempfile.gettempdir()):
    for idx, (page_num, page_drawings) in enumerate(drawings.items()):
        image = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(image)
    
        for drawing in page_drawings:
            min_x = float("inf")
            max_x = 0
            min_y = float("inf")
            max_y = 0

            for path in drawing["items"]:
                if path[0] == "l":  # Line segment
                    (x0, y0), (x1, y1) = path[1:]
                    draw.line((x0, y0, x1, y1), fill="black", width=2)
                    min_x = min(min_x, x0, x1)
                    max_x = max(max_x, x0, x1)
                    min_y = min(min_y, y0, y1)
                    max_y = max(max_y, y0, y1)
                elif path[0] == "re":  # Rectangle
                    x0, y0, x1, y1 = path[1]
                    draw.rectangle((x0, y0, x1, y1), outline="black")
                    min_x = min(min_x, x0, x1)
                    max_x = max(max_x, x0, x1)
                    min_y = min(min_y, y0, y1)
                    max_y = max(max_y, y0, y1)
                elif path[0] == "c":  # Curve (quadratic Bezier)
                    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = path[1:]
                    min_x = min(min_x, x0)
                    max_x = max(max_x, x2, x3)
                    min_y = min(min_y, y0, y1, y2, y3)
                    max_y = max(max_y, y0, y1, y2, y3)
                    draw.line((x0, y0, x1, y1, x2, y2, x3, y3), fill="black", width=2)
        crop_bbox = (min_x, min_y, max_x, max_y)
        cropped_image = image.crop(crop_bbox)
        if (cropped_image.size[0] != 0) and (cropped_image.size[1] != 0):
            img_name = f"{page_num}_{idx}.jpeg"
            img_path = os.path.join(dest_dir, img_name)
            cropped_image.save(img_path)
            yield img_path, crop_bbox


def add_images_redaction(pdf, images_text):
    nlp = load_nlp_model()

    for img_path, img_info in images_text.items():
        page_num, img_xref = (int(i) for i in Path(img_path).stem.split("_"))
        img_rect = pdf[page_num].get_image_rects(img_xref)[0]

        for line in img_info["text_lines"]:
            doc = nlp(line["text"])

            if doc.ents:
                width_scale = img_rect.width / img_info["image_size"][0]
                height_scale = img_rect.height / img_info["image_size"][1]
                text_rect = [ 
                    img_rect[0] + line["bbox"][0] * width_scale,
                    img_rect[1] + line["bbox"][1] * height_scale,
                    img_rect[0] + line["bbox"][2] * width_scale,
                    img_rect[1] + line["bbox"][3] * height_scale
                ]
                pdf[page_num].add_redact_annot(text_rect, fill=(0, 0, 0))


def add_drawings_redaction(pdf, analysis_result, drawings_images_info):
    for img_path, is_handwritten_signature in analysis_result:
        if is_handwritten_signature:
            page_num = int(Path(img_path).name.split("_")[0])
            drawing_bbox = [d for d in drawings_images_info if d[0] == img_path][0][1]
            pdf[page_num].add_redact_annot(drawing_bbox, fill=(0, 0, 0))


def main(input_file, output_file=os.path.join(tempfile.gettempdir(), "test.pdf"), reconstruct=True):
    input_file_path = Path(input_file)
    pdf = fitz.open(input_file_path.absolute().as_posix())
    #with Pool(processes=4) as pool:
    #    results = pool.starmap(analyze_print, (img_paths))
    #    for (generated_text, scores) in results:
    #        print(f"generated text: {generated_text}")
    paragraphs = extract_pdf_text(pdf)
    lines_subs = run_nlp(paragraphs)
    replacements = substitute_page_entities(pdf, lines_subs)
    if reconstruct:
        reconstruct_deleted_text(pdf, replacements, paragraphs)
    images = extract_pdf_images(pdf)
    img_paths = list(save_images(images))
    images_text = analyze_print(img_paths)
    add_images_redaction(pdf, images_text)
    drawings = extract_pdf_drawings(pdf)
    drawings_images_info = list(save_drawings(drawings, int(pdf[0].rect.width), int(pdf[0].rect.height)))
    drawing_images_paths = [d[0] for d in drawings_images_info]
    results = analyze_handwritting(drawing_images_paths)
    add_drawings_redaction(pdf, results, drawings_images_info)
    for page in pdf:
        page.apply_redactions()
    pdf.subset_fonts()
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.ez_save(output_file_path.absolute().as_posix())
    pdf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str),
    parser.add_argument("-r", "--reconstruct", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    main(args.input, args.output, args.reconstruct)