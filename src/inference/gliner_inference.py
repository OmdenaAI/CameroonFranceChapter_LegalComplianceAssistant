import fitz
import gliner
from utils.gliner_utils import resolve_overlaps_and_errors
from utils.gliner_utils import find_text_block
import settings

def replace_text_with_labels(model, input_path):
    doc = fitz.open(input_path)
    fontname = settings.FONT_NAME

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text = page.get_text()
        
        # Predict entities and filter overlaps
        entities = model.predict_entities(full_text, 
                                          labels=settings.LABELS, 
                                          threshold=settings.THRESHOLD)
        entities = resolve_overlaps_and_errors(entities)  # <-- New overlap resolution

        # Sort entities by position (reverse order to avoid coordinate shifts)
        entities.sort(key=lambda x: x["start"], reverse=True)

        # Replace entities in the PDF
        for entity in entities:
            original_text = entity["text"]
            label = entity["label"]

            # Find all occurrences of the entity on the page
            text_instances = page.search_for(original_text)

            for inst in text_instances:
                # Get original text properties
                block = find_text_block(page.get_text("dict"), inst) # real size of the original text
                if not block:
                    continue

                # Replace text with label using standard font
                # while adding redacted text, the original size is maintained to get original format
                # Example: [Person] is only 8 characters including square brackets
                # Let the name be Srihari Mohan which is 13 characters. so while redacting, 
                # '[Person]     ' this format is used
                page.add_redact_annot(
                    inst,
                    text=f"[{label}]",
                    fontsize=block["size"],
                    fontname=fontname,
                    text_color=block["color"],
                    fill=(1, 1, 1)
                )

        # Apply all redactions on the page
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return doc

