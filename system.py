import spacy
import calamancy
import tkinter as tk
from tkinter import ttk, scrolledtext
from entity_ruler_config import add_entity_ruler, map_cidoc_entity

def launch_gui():
    # Load your models
    loaded_models = {
        "Spacy Base": spacy.load("en_core_web_sm"),
        "Spacy Enhanced": spacy.load("trained_spacy_model"),
        "WikiAnn (Flaire) Base": spacy.load("xx_ent_wiki_sm"),
        "WikiAnn (Flaire) Enhanced": spacy.load("trained_multi_spacy_model"),
        "calamanCy MD Base": calamancy.load("tl_calamancy_md-0.1.0"),
        "calamanCy MD Enhanced": spacy.load("trained_calamanCy_medium_model"),
        "calamanCy LG": calamancy.load("tl_calamancy_lg-0.1.0"),
        "calamanCy LG Enhanced": spacy.load("trained_calamanCy_large_model"),
        # "calamanCy LG Enhanced (with CIDOC)": spacy.load("tl_calamancy_lg-v0.1.1"),
    }

    # --- Setup GUI ---
    root = tk.Tk()
    root.title("NER Model Comparison")
    root.configure(bg="#2c3e50")
    root.geometry("1200x650")
    root.minsize(900, 500)

    # Grid config
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("TLabel", background="#2c3e50", foreground="white", font=("Segoe UI", 10))
    style.configure("TButton", background="#1abc9c", foreground="white", font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("active", "#16a085")])
    style.configure("TCombobox", padding=5, font=("Segoe UI", 10))

    # Shared text input
    input_frame = ttk.Frame(root, padding=10)
    input_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
    input_frame.columnconfigure(0, weight=1)
    ttk.Label(input_frame, text="Input Text:").grid(row=0, column=0, sticky="w")
    input_text = tk.Text(input_frame, height=5, font=("Segoe UI", 10), wrap="word")
    input_text.insert(tk.END, "Enter your text here.")
    input_text.grid(row=1, column=0, sticky="ew", pady=5)

    # Evaluate button
    eval_button = ttk.Button(root, text="Evaluate", command=lambda: evaluate_text())
    eval_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))

    # Left model panel
    left_frame = ttk.LabelFrame(root, text="Model A", padding=10)
    left_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=10)
    root.grid_rowconfigure(2, weight=1)
    left_frame.columnconfigure(0, weight=1)
    model_a_dropdown = ttk.Combobox(left_frame, values=list(loaded_models.keys()), state="readonly")
    model_a_dropdown.current(0)
    model_a_dropdown.grid(row=0, column=0, sticky="ew", pady=5)
    output_a = scrolledtext.ScrolledText(left_frame, wrap="word", font=("Segoe UI", 10), bg="#ecf0f1", fg="#2c3e50")
    output_a.grid(row=1, column=0, sticky="nsew", pady=5)
    left_frame.rowconfigure(1, weight=1)

    # Right model panel
    right_frame = ttk.LabelFrame(root, text="Model B", padding=10)
    right_frame.grid(row=2, column=1, sticky="nsew", padx=(5, 10), pady=10)
    right_frame.columnconfigure(0, weight=1)
    model_b_dropdown = ttk.Combobox(right_frame, values=list(loaded_models.keys()), state="readonly")
    model_b_dropdown.current(1)
    model_b_dropdown.grid(row=0, column=0, sticky="ew", pady=5)
    output_b = scrolledtext.ScrolledText(right_frame, wrap="word", font=("Segoe UI", 10), bg="#ecf0f1", fg="#2c3e50")
    output_b.grid(row=1, column=0, sticky="nsew", pady=5)
    right_frame.rowconfigure(1, weight=1)

    # Evaluation logic
    def evaluate_text():
        text = input_text.get("1.0", tk.END).strip()
        model_a = loaded_models[model_a_dropdown.get()]
        model_b = loaded_models[model_b_dropdown.get()]
        doc_a = model_a(text)
        doc_b = model_b(text)

        # Output for model A
        output_a.delete("1.0", tk.END)
        output_a.insert(tk.END, "Predicted Entities:\n")
        if doc_a.ents:
            for ent in doc_a.ents:
                output_a.insert(tk.END, f"{ent.text} ({ent.label_})\n")
                # if ent.label_ == "CUL":
                #     cidoc_class = map_cidoc_entity(ent.text)
                #     output_a.insert(tk.END, f"{ent.text} ({ent.label_})-> {cidoc_class}\n")
                # else:
                #     output_a.insert(tk.END, f"{ent.text} ({ent.label_})\n")
        else:
            output_a.insert(tk.END, "No entities found.\n")

        # Entity Type Frequency - Model A
        entity_counts_a = {}
        for ent in doc_a.ents:
            entity_counts_a[ent.label_] = entity_counts_a.get(ent.label_, 0) + 1
        output_a.insert(tk.END, "\nEntity Type Frequency:\n")
        for label, count in entity_counts_a.items():
            output_a.insert(tk.END, f"{label}: {count}\n")

        # Entity Span Details - Model A
        output_a.insert(tk.END, "\nEntity Span Details:\n")
        for ent in doc_a.ents:
            output_a.insert(tk.END, f"Entity: {ent.text} | Start: {ent.start_char} | End: {ent.end_char}\n")

        # Output for model B
        output_b.delete("1.0", tk.END)
        output_b.insert(tk.END, "Predicted Entities:\n")
        if doc_b.ents:
            for ent in doc_b.ents:
                output_b.insert(tk.END, f"{ent.text} ({ent.label_})\n")
                # if ent.label_ == "CUL":
                #     cidoc_class = map_cidoc_entity(ent.text)
                #     output_b.insert(tk.END, f"{ent.text} ({ent.label_})-> {cidoc_class}\n")
                # else:
                #     output_b.insert(tk.END, f"{ent.text} ({ent.label_})\n")     
        else:
            output_b.insert(tk.END, "No entities found.\n")
        
        # Entity Type Frequency - Model B
        entity_counts_b = {}
        for ent in doc_b.ents:
            entity_counts_b[ent.label_] = entity_counts_b.get(ent.label_, 0) + 1
        output_b.insert(tk.END, "\nEntity Type Frequency:\n")
        for label, count in entity_counts_b.items():
            output_b.insert(tk.END, f"{label}: {count}\n")

        # Entity Span Details - - Model B
        output_b.insert(tk.END, "\nEntity Span Details:\n")
        for ent in doc_b.ents:
            output_b.insert(tk.END, f"Entity: {ent.text} | Start: {ent.start_char} | End: {ent.end_char}\n")


    root.mainloop()

if __name__ == "__main__":
    launch_gui()