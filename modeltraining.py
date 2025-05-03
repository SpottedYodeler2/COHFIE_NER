import spacy
import json
import random
import calamancy
from spacy.training.example import Example

def Model_Train(train_data, n_iter=20):
    """
    Train multiple spaCy models with the provided training data.

    Args:
        train_data (list): A list of tuples containing text and annotations.
        n_iter (int): Number of training iterations or epochs.

    Returns:
        dict: A dictionary containing training metrics for each model.
    """
    # Load models to be trained
    trained_spacy = spacy.load("en_core_web_sm")
    trained_multi_spacy = spacy.load("xx_ent_wiki_sm")
    trained_calamanCy_medium = calamancy.load("tl_calamancy_md-0.1.0")
    trained_calamanCy_large = calamancy.load("tl_calamancy_lg-0.1.0")

    # Dictionary of spaCy models to be trained
    spaCy_models = {
        "trained_spacy": trained_spacy,
        "trained_multi_spacy": trained_multi_spacy,
        "trained_calamanCy_medium": trained_calamanCy_medium,
        "trained_calamanCy_large": trained_calamanCy_large
    }

    # Add new NER labels from training data to each model
    for name, nlp_model in spaCy_models.items():
        if "ner" not in nlp_model.pipe_names:
            ner = nlp_model.add_pipe("ner")
        else:
            ner = nlp_model.get_pipe("ner")
        for text, annotations in train_data:
            for start, end, label in annotations.get("entities", []):
                ner.add_label(label)

    # Dictionary to hold metrics for each model
    training_metrics = {name: {'losses': []} for name in spaCy_models.keys()}

    # Train each spaCy model
    for name, nlp_model in spaCy_models.items():
        print(f"\nTraining model: {name}")
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

    # Save trained models and metrics to disk
    for name, nlp_model in spaCy_models.items():
        save_path = f"{name}_model"
        nlp_model.to_disk(save_path)
        print(f"Model '{name}' saved to {save_path}")

        metrics_path = f"{save_path}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics[name], f)
        print(f"Training metrics for '{name}' saved to {metrics_path}")

    return training_metrics
