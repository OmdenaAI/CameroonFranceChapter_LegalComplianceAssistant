import re
import spacy
import string
import consts
from spacy.tokens import Span
from spacy.language import Language


text = """
Common tasks for an IP address include both the identification of a host or a network, or identifying the location of a device. 
An IP address is not random. The creation of an IP address has the basis of math. 
The Internet Assigned Numbers Authority (IANA) allocates the IP address and its creation. 
The full range of IP addresses can go from 0.0.0.0 to 255.255.255.255. There are also IPv6 addresses supported.
Their range goes from ::2:3:4:5:6:7:8 to ::2:3:4:5:9:9:9. If you have any questions please contact our tech support either 
at techsupport@gmail.com or at tel. numbers +1 718 222 2222 or at +33 1 09 75 83 51. For the lastest information
please visit our web site at https://company.com.
"""

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

ENTITY_DESC = {
    "ORG": ("Company", string),
    "LOC": ("Location", string),
    "PERS": ("Person", string),
    "IPv4": ("IPv4 Address", int),
    "IPv6": ("IPv6 Address", int),
    "PHONE": ("Phone Number", int),
    "EMAIL": ("Email Address", int),
    "SSN": ("Social Security Number", int),
    "MEDICARE": ("Medicare Number", int),
    "VIN": ("Vehicle Identification Number", int),
    "URL": ("URL Address", int)
}


def find_matching_entities(doc, regex, label):
    matches = regex.finditer(doc.text)
    spans = [doc.char_span(match.start(), match.end(), label=label) for match in matches]
    doc.ents = list(doc.ents) + spans
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


def get_entities_subs(ents):
    subs = {}

    for ent in ents:
        ent_cat_subs = subs.get(ent.label_, {})
        ent_sub = ent_cat_subs.get(ent.text)
        if ent_sub is None:
            suffix_type = ENTITY_DESC[ent.label_][1]
            if suffix_type is string:
                ent_cat_subs[ent.text] = f"{ENTITY_DESC[ent.label_][0]} {string.ascii_uppercase[len(ent_cat_subs)]}"
            elif suffix_type is int:
                ent_cat_subs[ent.text] = f"{ENTITY_DESC[ent.label_][0]} {len(ent_cat_subs)+1}"
            else:
                raise Exception(f"Unsupported suffix type '{suffix_type}")
        subs[ent.label_] = ent_cat_subs

    return subs


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    nlp_acronyms = spacy.load(consts.ACRONYMS_MODEL_DIR)

    components = ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]

    for c in components:
        nlp.add_pipe(c, last=True)

    doc = nlp(text)
    acronyms_doc = nlp_acronyms(text)

    for ent in doc.ents:
        print(ent.text, ent.label_)

    subs = get_entities_subs(doc.ents)
    new_text = text

    for ent_type in list(subs.values()):
        for old_val, new_val in ent_type.items():
            new_text = new_text.replace(old_val, f"'{new_val}'")

    print(new_text)