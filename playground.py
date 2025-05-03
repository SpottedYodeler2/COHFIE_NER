import calamancy
import spacy
from entity_ruler_config import add_entity_ruler, map_cidoc_entity


if __name__ == "__main__":
    #Reload the test scripts for testing
    nlp_updated = spacy.load("tl_calamancy_lg-v0.1.1")

    # For inference, map a CUL entity to its CIDOC-CRM class:
    doc = nlp_updated("Pasko at Sinulog Festival ay mga mahalagang kaganapan sa kultura.")
    for ent in doc.ents:
        if ent.label_ == "CUL":
            cidoc_class = map_cidoc_entity(ent.text)
            print(f"{ent.text} -> {cidoc_class}")

    
