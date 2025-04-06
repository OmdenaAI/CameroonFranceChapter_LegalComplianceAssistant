import fitz
import settings
def resolve_overlaps_and_errors(entities):  # to manage ties between labels
    """Resolve overlaps AND filter incorrect labels like 'marriage' → [State]"""
    priority_labels = settings.PRIORITY_LABELS
    resolved = []

    # Sort by length (longest first) to prioritize broader matches
    entities.sort(key=lambda x: x["end"] - x["start"], reverse=True)

    for entity in entities:
        text = entity["text"].lower()  # Case-insensitive check

        # --- Custom Rule to Fix "marriage" → [State] ---
        if entity["label"] == "State" and "Address" in text:
            continue  # Skip this entity

        # --- Custom Rule to Fix "ia" → [State] ---
        if entity["label"] == "State" and text == "ia":
            continue  # Skip standalone "ia" (if not part of a valid context)

        # --- Standard Overlap Resolution ---
        is_contained = False
        for resolved_entity in resolved:
            if (entity["start"] >= resolved_entity["start"] and
                entity["end"] <= resolved_entity["end"]):
                is_contained = True
                break

        if not is_contained:
            resolved.append(entity)

    return resolved

def find_text_block(text_dict, rect):
    """Find text block containing the given rectangle"""
    for block in text_dict["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                if span_rect.intersects(rect):
                    return span
    return None