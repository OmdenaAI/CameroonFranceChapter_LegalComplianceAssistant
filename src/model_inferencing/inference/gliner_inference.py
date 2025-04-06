import fitz
import gliner
from utils.gliner_utils import resolve_overlaps_and_errors
from utils.gliner_utils import find_text_block
import settings

def split_text_into_chunks(text, max_tokens=296):
    
    words = text.split()  # Split text into words
    chunks = []
    current_chunk = []

    for word in words:
        # Check if adding the next word exceeds the token limit (approximation)
        if len(" ".join(current_chunk + [word]).split()) <= max_tokens:
            current_chunk.append(word)
        else:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def replace_text_with_labels(model, input_path):
    print(input_path)
    doc = fitz.open(input_path)
    fontname = settings.FONT_NAME
    new_entities = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text = page.get_text()

        # Step 1: Split the full text into chunks
        chunks = split_text_into_chunks(full_text, max_tokens=296)

        # Step 2: Process each chunk and aggregate entities
        all_entities = []
        for chunk in chunks:
            # Predict entities for the current chunk
            entities = model.predict_entities(
                chunk,
                labels=settings.LABELS,
                threshold=settings.THRESHOLD
            )
            all_entities.extend(entities)

        # Resolve overlaps and errors across all entities from all chunks
        all_entities = resolve_overlaps_and_errors(all_entities)

        # Sort entities by position (reverse order to avoid coordinate shifts)
        all_entities.sort(key=lambda x: x["start"], reverse=True)

        # Differentiate values
        for ent in all_entities:
            if ent['text'] not in new_entities.values():
                # Find all existing keys with the same label prefix
                match_keys = [
                    int(k.split("__")[1]) for k in new_entities.keys()
                    if k.split("__")[0] == ent['label']
                ]
                
                # Determine the next suffix (increment the largest suffix by 1, or start from 0 if no matches)
                new_suffix = max(match_keys, default=-1) + 1
                
                # Add the new entity with the updated suffix
                new_entities[f"{ent['label']}__{new_suffix}"] = ent['text']

        # Replace entities in the PDF
        for entity in all_entities:
            original_text = entity["text"]
            for key, value in new_entities.items():
                if value == original_text:
                    label = key  # Update the label

            # Find all occurrences of the entity on the page
            text_instances = page.search_for(original_text)

            for inst in text_instances:
                # Get original text properties
                block = find_text_block(page.get_text("dict"), inst)  # Real size of the original text
                if not block:
                    continue

                # Replace text with label using standard font
                page.add_redact_annot(
                    inst,
                    text=f"[{label}]",
                    fontsize=block["size"],
                    fontname=fontname,
                    fill=(1, 1, 1)
                )

        # Apply all redactions on the page
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    return doc