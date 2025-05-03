import json
import preprocessing as prep
import spacy
import random
import calamancy

from spacy.training.example import Example

if __name__ == "__main__":

    with open("Datasets/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # trained_spacy = spacy.load("en_core_web_sm")
    # trained_multi_spacy = spacy.load("xx_ent_wiki_sm")
    # trained_calamanCy_medium = calamancy.load("tl_calamancy_md-0.1.0")
    trained_calamanCy_large  = calamancy.load("tl_calamancy_lg-0.1.0")

    # Dictionary of spaCy models to be trained.
    spaCy_models = {
        # "trained_spacy": trained_spacy,
        # "trained_multi_spacy": trained_multi_spacy,
        # "trained_calamanCy_medium": trained_calamanCy_medium,
        "trained_calamanCy_large": trained_calamanCy_large
    }

    # For each spaCy model, add new NER labels from training data.
    for name, nlp_model in spaCy_models.items():
        if "ner" not in nlp_model.pipe_names:
            ner = nlp_model.add_pipe("ner")
        else:
            ner = nlp_model.get_pipe("ner")
        # Add all entity labels present in the training data
        for text, annotations in train_data:
            for start, end, label in annotations.get("entities", []):
                ner.add_label(label)
    
    
    # Dictionary to hold metrics for each model
    training_metrics = {name: {'losses': []} for name in spaCy_models.keys()}   

    # -----------------------------------------------------------------------------
    # Train Each spaCy Model
    # -----------------------------------------------------------------------------
    n_iter = 20  # Number of training iterations or epochs

    for name, nlp_model in spaCy_models.items():
        print(f"\nTraining model: {name}")
        # Disable all pipes except NER during training
        other_pipes = [pipe for pipe in nlp_model.pipe_names if pipe != "ner"]
        with nlp_model.disable_pipes(*other_pipes):
            optimizer = nlp_model.resume_training()
            for itn in range(n_iter):
                losses = {}
                random.shuffle(train_data)
                for text, annotations in train_data:
                    doc = nlp_model.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp_model.update([example], sgd=optimizer, drop=0.35, losses=losses)

                # Store and print the loss for this iteration
                current_loss = float(losses.get('ner', 0.0))
                training_metrics[name]['losses'].append(current_loss)
                print(f"Iteration {itn+1}/{n_iter} - Loss: {current_loss:.4f}")

    # -----------------------------------------------------------------------------
    # Save Trained Models to Disk (for later use in Colab)
    # -----------------------------------------------------------------------------
    # Save trained models in Google Colab
    for name, nlp_model in spaCy_models.items():
        save_path = f"{name}_model"
        nlp_model.to_disk(save_path)
        print(f"Model '{name}' saved to {save_path}")

        # Save training metrics
        metrics_path = f"{save_path}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics[name], f)
        print(f"Training metrics for '{name}' saved to {metrics_path}")



