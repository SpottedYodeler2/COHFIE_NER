# entity_ruler_config.py

from spacy.pipeline import EntityRuler

def add_entity_ruler(nlp):
    """
    Adds an EntityRuler with predefined patterns for CUL, PER, ORG, and LOC
    based on the TRAIN_DATA examples.
    """
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        # ---------- CULTURAL HERITAGE (CUL) patterns ----------
        {"label": "CUL", "pattern": "Pasko"},
        {"label": "CUL", "pattern": [{"LOWER": "sinulog"}, {"LOWER": "festival"}]},
        {"label": "CUL", "pattern": "Banaue Rice Terraces"},
        {"label": "CUL", "pattern": [{"LOWER": "ifugao"}]},
        {"label": "CUL", "pattern": [{"LOWER": "igorot"}]},
        {"label": "CUL", "pattern": "Ifugao"},
        {"label": "CUL", "pattern": "Igorot"},
        {"label": "CUL", "pattern": "Apayao"},
        {"label": "CUL", "pattern": "Kalinga"},
        {"label": "CUL", "pattern": "Amerikano"},
        {"label": "CUL", "pattern": "Bayanihan"},
        {"label": "CUL", "pattern": "Sinulog Festival"},
        {"label": "CUL", "pattern": "Balagtasan"},
        {"label": "CUL", "pattern": "Pahiyas Festival"},
        {"label": "CUL", "pattern": "Ati-Atihan Festival"},
        {"label": "CUL", "pattern": "Bahay Kubo"},
        {"label": "CUL", "pattern": "Kundiman"},
        {"label": "CUL", "pattern": "Tinikling"},
        {"label": "CUL", "pattern": "Jeepney"},
        {"label": "CUL", "pattern": "Barong Tagalog"},
        {"label": "CUL", "pattern": "Tradisyunal"},
        {"label": "CUL", "pattern": "Sining"},
        {"label": "CUL", "pattern": "Tsino"},
        {"label": "CUL", "pattern": "Batangueño"},
        {"label": "CUL", "pattern": "Kultura"},
        {"label": "CUL", "pattern": "Hapon"},

        # ---------- PERSON (PER) patterns ----------
        {"label": "PER", "pattern": "Otley Beyer"},
        {"label": "PER", "pattern": "AnAlizA D. ResuRReccion"},
        {"label": "PER", "pattern": "Henry Otley Beyer"},
        {"label": "PER", "pattern": "Beyer"},
        {"label": "PER", "pattern": "Francis Burton Harrison"},
        {"label": "PER", "pattern": "Governor Harrison"},
        {"label": "PER", "pattern": "Fuji"},
        {"label": "PER", "pattern": "Percy Hill"},
        {"label": "PER", "pattern": "Jose Caedo"},
        {"label": "PER", "pattern": "Tarcila Malabanan"},
        {"label": "PER", "pattern": "Cornelio V. Umali"},
        {"label": "PER", "pattern": "Encarnacion R. Buendia"},
        {"label": "PER", "pattern": "Lorenzo Brotenel"},
        {"label": "PER", "pattern": "Brotonel"},
        {"label": "PER", "pattern": "Manuel L. Quezon"},
        {"label": "PER", "pattern": "Juan dela Cruz"},
        {"label": "PER", "pattern": "Andres Bonifacio"},
        {"label": "PER", "pattern": "Corazon Aquino"},
        {"label": "PER", "pattern": "Jose Rizal"},
        {"label": "PER", "pattern": "Lualhati Bautista"},
        {"label": "PER", "pattern": "Francisco Baltazar"},
        {"label": "PER", "pattern": "Nick Joaquin"},
        {"label": "PER", "pattern": "Liza Soberano"},
        {"label": "PER", "pattern": "Manny Pacquiao"},
        {"label": "PER", "pattern": "Lea Salonga"},
        {"label": "PER", "pattern": "Efren Reyes"},
        {"label": "PER", "pattern": "Charice"},
        {"label": "PER", "pattern": "Maria Clara"},
        {"label": "PER", "pattern": "Apolinario Mabini"},

        # ---------- ORGANIZATION (ORG) patterns ----------
        {"label": "ORG", "pattern": "PambansANG Aklatan ng Pilipinas"},
        {"label": "ORG", "pattern": "KEywOrDs Beyer Collection"},
        {"label": "ORG", "pattern": "Denver University"},
        {"label": "ORG", "pattern": "Harvard University"},
        {"label": "ORG", "pattern": "Bureau of Science"},
        {"label": "ORG", "pattern": "PaNIMULa A. Kaligiran ng Pag-aaral"},
        {"label": "ORG", "pattern": "Unibersidad ng Pilipinas"},
        {"label": "ORG", "pattern": "Kagawaran ng Antropolohiya"},
        {"label": "ORG", "pattern": "Museum and Institute of Archaeology & Ethnology, Manila"},
        {"label": "ORG", "pattern": "Bureau of Census and Statistics"},
        {"label": "ORG", "pattern": "Journal of American Chamber of Commerce of the Philippines"},
        {"label": "ORG", "pattern": "University of Michigan"},
        {"label": "ORG", "pattern": "De La Salle Lipa- Center for Batangas Studies"},
        {"label": "ORG", "pattern": "Philippine Commission"},
        {"label": "ORG", "pattern": "Meralco"},
        {"label": "ORG", "pattern": "Kalinga Cultural Preservation Council"},
        {"label": "ORG", "pattern": "Cultural Center of the Philippines"},
        {"label": "ORG", "pattern": "National Commission for Culture and the Arts"},
        {"label": "ORG", "pattern": "National Historical Commission of the Philippines"},
        {"label": "ORG", "pattern": "Sining at Kultura ng Mindanao Foundation"},
        {"label": "ORG", "pattern": "Department of Tourism ng Pilipinas"},
        {"label": "ORG", "pattern": "Ayala Foundation"},
        {"label": "ORG", "pattern": "Philippine Business for Social Progress"},
        {"label": "ORG", "pattern": "Bayanihan Cooperative"},
        {"label": "ORG", "pattern": "Department of Education ng Pilipinas"},
        {"label": "ORG", "pattern": "Pilipinas Cultural Forum"},
        {"label": "ORG", "pattern": "National Museum of the Philippines"},
        {"label": "ORG", "pattern": "Arts Embroidery Company"},
        {"label": "ORG", "pattern": "Batangas Minerals and Consolidated Mines"},
        {"label": "ORG", "pattern": "Journal of Philippine Chamber of Commerce"},
        {"label": "ORG", "pattern": "A Pronouncing Gazetteer and Geographical Dictionary of the Philippine Islands"},

        # ---------- LOCATION (LOC) patterns ----------
        {"label": "LOC", "pattern": "Batangas"},
        {"label": "LOC", "pattern": "Pilipinas"},
        {"label": "LOC", "pattern": "Edgewood, Iowa"},
        {"label": "LOC", "pattern": "Banaue Valley"},
        {"label": "LOC", "pattern": "Luzon"},
        {"label": "LOC", "pattern": "Asya"},
        {"label": "LOC", "pattern": "Hilagang Africa"},
        {"label": "LOC", "pattern": "Europa"},
        {"label": "LOC", "pattern": "San Pablo"},
        {"label": "LOC", "pattern": "Laguna"},
        {"label": "LOC", "pattern": "Taal"},
        {"label": "LOC", "pattern": "Vigan"},
        {"label": "LOC", "pattern": "Ilocos Sur"},
        {"label": "LOC", "pattern": "Ermita"},
        {"label": "LOC", "pattern": "Maynila"},
        {"label": "LOC", "pattern": "Kalye Jose P. Laurel"},
        {"label": "LOC", "pattern": "San Miguel"},
        {"label": "LOC", "pattern": "Kalye Nebraska"},
        {"label": "LOC", "pattern": "Kalye Jorge Bocobo"},
        {"label": "LOC", "pattern": "Balayan"},
        {"label": "LOC", "pattern": "Nasugbu"},
        {"label": "LOC", "pattern": "Cavite"},
        {"label": "LOC", "pattern": "Baguio"},
        {"label": "LOC", "pattern": "Lucban"},
        {"label": "LOC", "pattern": "Quezon"},
        {"label": "LOC", "pattern": "Cebu"},
        {"label": "LOC", "pattern": "Luneta"},
        {"label": "LOC", "pattern": "Sagada"},
        {"label": "LOC", "pattern": "Tagaytay"},
        {"label": "LOC", "pattern": "Santo Tomas"},
        {"label": "LOC", "pattern": "Tayabas"},
        {"label": "LOC", "pattern": "Rosario"},
        {"label": "LOC", "pattern": "Calamba"},
        {"label": "LOC", "pattern": "Bulacan"},
        {"label": "LOC", "pattern": "Rizal"},
        {"label": "LOC", "pattern": "Mountain Province"},
        {"label": "LOC", "pattern": "Negros Occidental"},
        {"label": "LOC", "pattern": "Negros Oriental"},
        {"label": "LOC", "pattern": "Panay"},
        {"label": "LOC", "pattern": "Tarlac"},
        {"label": "LOC", "pattern": "Capiz"},
        {"label": "LOC", "pattern": "Mindoro"},
        {"label": "LOC", "pattern": "Romblon"},
        {"label": "LOC", "pattern": "Calumpang"},
        {"label": "LOC", "pattern": "Muñoz, Nueva Ecija"},
        {"label": "LOC", "pattern": "Lungsod ng Batangas"},
        {"label": "LOC", "pattern": "National Capital Region"},
        {"label": "LOC", "pattern": "CALABARZON"},
        {"label": "CUL", "pattern": "Pasko"},
        {"label": "CUL", "pattern": [{"LOWER": "sinulog"}, {"LOWER": "festival"}]},
        {"label": "CUL", "pattern": "Banaue Rice Terraces"},
        {"label": "CUL", "pattern": [{"LOWER": "ifugao"}]},
        {"label": "CUL", "pattern": [{"LOWER": "igorot"}]},
        
        # ---------- PERSON (PER) patterns ----------
        {"label": "PER", "pattern": "Otley Beyer"},
        {"label": "PER", "pattern": "AnAlizA D. ResuRReccion"},
        {"label": "PER", "pattern": "Henry Otley Beyer"},
        {"label": "PER", "pattern": "Beyer"},
        {"label": "PER", "pattern": "Francis Burton Harrison"},
        {"label": "PER", "pattern": "Governor Harrison"},
        {"label": "PER", "pattern": "Fuji"},
        {"label": "PER", "pattern": "Percy Hill"},
        {"label": "PER", "pattern": "Jose Caedo"},
        {"label": "PER", "pattern": "Tarcila Malabanan"},
        {"label": "PER", "pattern": "Cornelio V. Umali"},
        {"label": "PER", "pattern": "Encarnacion R. Buendia"},
        {"label": "PER", "pattern": "Lorenzo Brotenel"},
        {"label": "PER", "pattern": "Brotonel"},
        {"label": "PER", "pattern": "Manuel L. Quezon"},
        {"label": "PER", "pattern": "Juan dela Cruz"},
        {"label": "PER", "pattern": "Andres Bonifacio"},
        {"label": "PER", "pattern": "Corazon Aquino"},
        {"label": "PER", "pattern": "Jose Rizal"},
        {"label": "PER", "pattern": "Lualhati Bautista"},
        {"label": "PER", "pattern": "Francisco Baltazar"},
        {"label": "PER", "pattern": "Nick Joaquin"},
        {"label": "PER", "pattern": "Liza Soberano"},
        {"label": "PER", "pattern": "Manny Pacquiao"},
        {"label": "PER", "pattern": "Lea Salonga"},
        {"label": "PER", "pattern": "Efren Reyes"},
        {"label": "PER", "pattern": "Charice"},
        {"label": "PER", "pattern": "Maria Clara"},
        {"label": "PER", "pattern": "Apolinario Mabini"},
        
        # ---------- ORGANIZATION (ORG) patterns ----------
        {"label": "ORG", "pattern": "PambansANG Aklatan ng Pilipinas"},
        {"label": "ORG", "pattern": "KEywOrDs Beyer Collection"},
        {"label": "ORG", "pattern": "Denver University"},
        {"label": "ORG", "pattern": "Harvard University"},
        {"label": "ORG", "pattern": "Bureau of Science"},
        {"label": "ORG", "pattern": "PaNIMULa A. Kaligiran ng Pag-aaral"},
        {"label": "ORG", "pattern": "Unibersidad ng Pilipinas"},
        {"label": "ORG", "pattern": "Kagawaran ng Antropolohiya"},
        {"label": "ORG", "pattern": "Museum and Institute of Archaeology & Ethnology, Manila"},
        {"label": "ORG", "pattern": "Bureau of Census and Statistics"},
        {"label": "ORG", "pattern": "Journal of American Chamber of Commerce of the Philippines"},
        {"label": "ORG", "pattern": "University of Michigan"},
        {"label": "ORG", "pattern": "De La Salle Lipa- Center for Batangas Studies"},
        {"label": "ORG", "pattern": "Philippine Commission"},
        {"label": "ORG", "pattern": "Meralco"},
        {"label": "ORG", "pattern": "Kalinga Cultural Preservation Council"},
        {"label": "ORG", "pattern": "Cultural Center of the Philippines"},
        {"label": "ORG", "pattern": "National Commission for Culture and the Arts"},
        {"label": "ORG", "pattern": "National Historical Commission of the Philippines"},
        {"label": "ORG", "pattern": "Sining at Kultura ng Mindanao Foundation"},
        {"label": "ORG", "pattern": "Department of Tourism ng Pilipinas"},
        {"label": "ORG", "pattern": "Ayala Foundation"},
        {"label": "ORG", "pattern": "Philippine Business for Social Progress"},
        {"label": "ORG", "pattern": "Bayanihan Cooperative"},
        {"label": "ORG", "pattern": "Department of Education ng Pilipinas"},
        {"label": "ORG", "pattern": "Pilipinas Cultural Forum"},
        {"label": "ORG", "pattern": "National Museum of the Philippines"},
        
        # ---------- LOCATION (LOC) patterns ----------
        {"label": "LOC", "pattern": "Batangas"},
        {"label": "LOC", "pattern": "Pilipinas"},
        {"label": "LOC", "pattern": "Edgewood, Iowa"},
        {"label": "LOC", "pattern": "Banaue Valley"},
        {"label": "LOC", "pattern": "Luzon"},
        {"label": "LOC", "pattern": "Asya"},
        {"label": "LOC", "pattern": "Hilagang Africa"},
        {"label": "LOC", "pattern": "Europa"},
        {"label": "LOC", "pattern": "San Pablo"},
        {"label": "LOC", "pattern": "Laguna"},
        {"label": "LOC", "pattern": "Taal"},
        {"label": "LOC", "pattern": "Vigan"},
        {"label": "LOC", "pattern": "Ilocos Sur"},
        {"label": "LOC", "pattern": "Ermita"},
        {"label": "LOC", "pattern": "Maynila"},
        {"label": "LOC", "pattern": "Kalye Jose P. Laurel"},
        {"label": "LOC", "pattern": "San Miguel"},
        {"label": "LOC", "pattern": "Kalye Nebraska"},
        {"label": "LOC", "pattern": "Kalye Jorge Bocobo"},
        {"label": "LOC", "pattern": "Balayan"},
        {"label": "LOC", "pattern": "Nasugbu"},
        {"label": "LOC", "pattern": "Cavite"},
        {"label": "LOC", "pattern": "Baguio"},
        {"label": "LOC", "pattern": "Lucban"},
        {"label": "LOC", "pattern": "Quezon"},
        {"label": "LOC", "pattern": "Cebu"},
        {"label": "LOC", "pattern": "Luneta"},
        {"label": "LOC", "pattern": "Sagada"},
        {"label": "LOC", "pattern": "Tagaytay"},
        {"label": "LOC", "pattern": "Santo Tomas"},
        {"label": "LOC", "pattern": "Tayabas"},
        {"label": "LOC", "pattern": "Rosario"},
        {"label": "LOC", "pattern": "Calamba"},
        {"label": "LOC", "pattern": "Bulacan"},
        {"label": "LOC", "pattern": "Rizal"},
        {"label": "LOC", "pattern": "Mountain Province"},
        {"label": "LOC", "pattern": "Negros Occidental"},
        {"label": "LOC", "pattern": "Negros Oriental"},
        {"label": "LOC", "pattern": "Panay"},
        {"label": "LOC", "pattern": "Tarlac"},
        {"label": "LOC", "pattern": "Capiz"},
        {"label": "LOC", "pattern": "Mindoro"},
        {"label": "LOC", "pattern": "Romblon"},
        {"label": "LOC", "pattern": "Calumpang"},
        {"label": "LOC", "pattern": "Muñoz, Nueva Ecija"},
        
        # ---------- CULTURAL HERITAGE (CUL) patterns ----------
        {"label": "CUL", "pattern": "Ifugao"},
        {"label": "CUL", "pattern": "Igorot"},
        {"label": "CUL", "pattern": "Apayao"},
        {"label": "CUL", "pattern": "Kalinga"},
        {"label": "CUL", "pattern": "Amerikano"},
        {"label": "CUL", "pattern": "Bayanihan"},
        {"label": "CUL", "pattern": "Pasko"},
        {"label": "CUL", "pattern": "Sinulog Festival"},
        {"label": "CUL", "pattern": "Balagtasan"},
        {"label": "CUL", "pattern": "Pahiyas Festival"},
        {"label": "CUL", "pattern": "Ati-Atihan Festival"},
        {"label": "CUL", "pattern": "Bahay Kubo"},
        {"label": "CUL", "pattern": "Kundiman"},
        {"label": "CUL", "pattern": "Tinikling"},
        {"label": "CUL", "pattern": "Jeepney"},
        {"label": "CUL", "pattern": "Barong Tagalog"},
        {"label": "CUL", "pattern": "Tradisyunal"},
        {"label": "CUL", "pattern": "Sining"},
        {"label": "CUL", "pattern": "Tsino"},
        {"label": "CUL", "pattern": "Batangueño"},
        {"label": "CUL", "pattern": "Kultura"},

        # ---------- PERSON (PER) patterns ----------
        {"label": "PER", "pattern": "Otley Beyer"},
        {"label": "PER", "pattern": "AnAlizA D. ResuRReccion"},
        {"label": "PER", "pattern": "Henry Otley Beyer"},
        {"label": "PER", "pattern": "Beyer"},
        {"label": "PER", "pattern": "Francis Burton Harrison"},
        {"label": "PER", "pattern": "Governor Harrison"},
        {"label": "PER", "pattern": "Fuji"},
        {"label": "PER", "pattern": "Percy Hill"},
        {"label": "PER", "pattern": "Jose Caedo"},
        {"label": "PER", "pattern": "Tarcila Malabanan"},
        {"label": "PER", "pattern": "Cornelio V. Umali"},
        {"label": "PER", "pattern": "Encarnacion R. Buendia"},
        {"label": "PER", "pattern": "Lorenzo Brotenel"},
        {"label": "PER", "pattern": "Brotonel"},
        {"label": "PER", "pattern": "Manuel L. Quezon"},
        {"label": "PER", "pattern": "Juan dela Cruz"},
        {"label": "PER", "pattern": "Andres Bonifacio"},
        {"label": "PER", "pattern": "Corazon Aquino"},
        {"label": "PER", "pattern": "Jose Rizal"},
        {"label": "PER", "pattern": "Lualhati Bautista"},
        {"label": "PER", "pattern": "Francisco Baltazar"},
        {"label": "PER", "pattern": "Nick Joaquin"},
        {"label": "PER", "pattern": "Liza Soberano"},
        {"label": "PER", "pattern": "Manny Pacquiao"},
        {"label": "PER", "pattern": "Lea Salonga"},
        {"label": "PER", "pattern": "Efren Reyes"},
        {"label": "PER", "pattern": "Charice"},
        {"label": "PER", "pattern": "Maria Clara"},
        {"label": "PER", "pattern": "Apolinario Mabini"},

        # ---------- ORGANIZATION (ORG) patterns ----------
        {"label": "ORG", "pattern": "PambansANG Aklatan ng Pilipinas"},
        {"label": "ORG", "pattern": "KEywOrDs Beyer Collection"},
        {"label": "ORG", "pattern": "Denver University"},
        {"label": "ORG", "pattern": "Harvard University"},
        {"label": "ORG", "pattern": "Bureau of Science"},
        {"label": "ORG", "pattern": "PaNIMULa A. Kaligiran ng Pag-aaral"},
        {"label": "ORG", "pattern": "Unibersidad ng Pilipinas"},
        {"label": "ORG", "pattern": "Kagawaran ng Antropolohiya"},
        {"label": "ORG", "pattern": "Museum and Institute of Archaeology & Ethnology, Manila"},
        {"label": "ORG", "pattern": "Bureau of Census and Statistics"},
        {"label": "ORG", "pattern": "Journal of American Chamber of Commerce of the Philippines"},
        {"label": "ORG", "pattern": "University of Michigan"},
        {"label": "ORG", "pattern": "De La Salle Lipa- Center for Batangas Studies"},
        {"label": "ORG", "pattern": "Philippine Commission"},
        {"label": "ORG", "pattern": "Meralco"},
        {"label": "ORG", "pattern": "Kalinga Cultural Preservation Council"},
        {"label": "ORG", "pattern": "Cultural Center of the Philippines"},
        {"label": "ORG", "pattern": "National Commission for Culture and the Arts"},
        {"label": "ORG", "pattern": "National Historical Commission of the Philippines"},
        {"label": "ORG", "pattern": "Sining at Kultura ng Mindanao Foundation"},
        {"label": "ORG", "pattern": "Department of Tourism ng Pilipinas"},
        {"label": "ORG", "pattern": "Ayala Foundation"},
        {"label": "ORG", "pattern": "Philippine Business for Social Progress"},
        {"label": "ORG", "pattern": "Bayanihan Cooperative"},
        {"label": "ORG", "pattern": "Department of Education ng Pilipinas"},
        {"label": "ORG", "pattern": "Pilipinas Cultural Forum"},
        {"label": "ORG", "pattern": "National Museum of the Philippines"},
        {"label": "ORG", "pattern": "Arts Embroidery Company"},
        {"label": "ORG", "pattern": "Batangas Minerals and Consolidated Mines"},
        {"label": "ORG", "pattern": "Journal of Philippine Chamber of Commerce"},
        {"label": "ORG", "pattern": "A Pronouncing Gazetteer and Geographical Dictionary of the Philippine Islands"},

        # ---------- LOCATION (LOC) patterns ----------
        {"label": "LOC", "pattern": "Batangas"},
        {"label": "LOC", "pattern": "Pilipinas"},
        {"label": "LOC", "pattern": "Edgewood, Iowa"},
        {"label": "LOC", "pattern": "Banaue Valley"},
        {"label": "LOC", "pattern": "Luzon"},
        {"label": "LOC", "pattern": "Asya"},
        {"label": "LOC", "pattern": "Hilagang Africa"},
        {"label": "LOC", "pattern": "Europa"},
        {"label": "LOC", "pattern": "San Pablo"},
        {"label": "LOC", "pattern": "Laguna"},
        {"label": "LOC", "pattern": "Taal"},
        {"label": "LOC", "pattern": "Vigan"},
        {"label": "LOC", "pattern": "Ilocos Sur"},
        {"label": "LOC", "pattern": "Ermita"},
        {"label": "LOC", "pattern": "Maynila"},
        {"label": "LOC", "pattern": "Kalye Jose P. Laurel"},
        {"label": "LOC", "pattern": "San Miguel"},
        {"label": "LOC", "pattern": "Kalye Nebraska"},
        {"label": "LOC", "pattern": "Kalye Jorge Bocobo"},
        {"label": "LOC", "pattern": "Balayan"},
        {"label": "LOC", "pattern": "Nasugbu"},
        {"label": "LOC", "pattern": "Cavite"},
        {"label": "LOC", "pattern": "Baguio"},
        {"label": "LOC", "pattern": "Lucban"},
        {"label": "LOC", "pattern": "Quezon"},
        {"label": "LOC", "pattern": "Cebu"},
        {"label": "LOC", "pattern": "Luneta"},
        {"label": "LOC", "pattern": "Sagada"},
        {"label": "LOC", "pattern": "Tagaytay"},
        {"label": "LOC", "pattern": "Santo Tomas"},
        {"label": "LOC", "pattern": "Tayabas"},
        {"label": "LOC", "pattern": "Rosario"},
        {"label": "LOC", "pattern": "Calamba"},
        {"label": "LOC", "pattern": "Bulacan"},
        {"label": "LOC", "pattern": "Rizal"},
        {"label": "LOC", "pattern": "Mountain Province"},
        {"label": "LOC", "pattern": "Negros Occidental"},
        {"label": "LOC", "pattern": "Negros Oriental"},
        {"label": "LOC", "pattern": "Panay"},
        {"label": "LOC", "pattern": "Tarlac"},
        {"label": "LOC", "pattern": "Capiz"},
        {"label": "LOC", "pattern": "Mindoro"},
        {"label": "LOC", "pattern": "Romblon"},
        {"label": "LOC", "pattern": "Calumpang"},
        {"label": "LOC", "pattern": "Muñoz, Nueva Ecija"},
        {"label": "LOC", "pattern": "Lungsod ng Batangas"},
        {"label": "LOC", "pattern": "National Capital Region"},
        {"label": "LOC", "pattern": "CALABARZON"},

        # ---------- CULTURAL HERITAGE (CUL) patterns ----------
        {"label": "CUL", "pattern": "Ifugao"},
        {"label": "CUL", "pattern": "Igorot"},
        {"label": "CUL", "pattern": "Apayao"},
        {"label": "CUL", "pattern": "Kalinga"},
        {"label": "CUL", "pattern": "Amerikano"},
        {"label": "CUL", "pattern": "Bayanihan"},
        {"label": "CUL", "pattern": "Pasko"},
        {"label": "CUL", "pattern": "Sinulog Festival"},
        {"label": "CUL", "pattern": "Balagtasan"},
        {"label": "CUL", "pattern": "Pahiyas Festival"},
        {"label": "CUL", "pattern": "Ati-Atihan Festival"},
        {"label": "CUL", "pattern": "Bahay Kubo"},
        {"label": "CUL", "pattern": "Kundiman"},
        {"label": "CUL", "pattern": "Tinikling"},
        {"label": "CUL", "pattern": "Jeepney"},
        {"label": "CUL", "pattern": "Barong Tagalog"},
        {"label": "CUL", "pattern": "Tradisyunal"},
        {"label": "CUL", "pattern": "Sining"},
        {"label": "CUL", "pattern": "Tsino"},
        {"label": "CUL", "pattern": "Batangueño"},
        {"label": "CUL", "pattern": "Kultura"},
        # Sometimes authority names like "Hapon" are tagged as cultural heritage:
        {"label": "CUL", "pattern": "Hapon"}

        # (Repeat some patterns if needed for maximum coverage)
    ]
    ruler.add_patterns(patterns)

###############################################################################
# Heuristic Mapping for CIDOC-CRM
###############################################################################
# This function maps a given entity text to one of the 20 CIDOC-CRM classes
# based on keywords. You can refine these rules as needed.
###############################################################################

def map_cidoc_entity(entity_text):
    """
    Map a given entity text (typically a CUL entity) to one of the 20 CIDOC-CRM
    classes using heuristic keyword matching.
    """
    lower_text = entity_text.lower()

    # E25 Man-Made Feature (immovable architectural features)
    if "rice terraces" in lower_text or "intramuros" in lower_text or "wall" in lower_text or "kuta" in lower_text:
        return "E25 Man-Made Feature"
    
    # E22 Man-Made Object (tangible artifacts)
    if "barong" in lower_text or "jeepney" in lower_text or "katipunan flag" in lower_text or "vinta" in lower_text:
        return "E22 Man-Made Object"
    
    # E78 Collection (groups of related objects)
    if "koleksyon" in lower_text or "collection" in lower_text:
        return "E78 Collection"
    
    # E5 Event (historical/cultural occurrences)
    if "misa de gallo" in lower_text or "cry of pugad lawin" in lower_text or "event" in lower_text:
        return "E5 Event"
    
    # E7 Activity (festivals, dances, rituals)
    if any(word in lower_text for word in ["festival", "dance", "ritual", "performance", "tinikling", "balagtasan", "pahiyas", "ati-atihan", "panagbenga", "pasko"]):
        return "E7 Activity"
    
    # E4 Period (historical epochs)
    if "colonial period" in lower_text or "occupation" in lower_text or "era" in lower_text:
        return "E4 Period"
    
    # E12 Production (artifact creation events)
    if "noli me tangere" in lower_text or "creation" in lower_text:
        return "E12 Production"
    
    # E8 Acquisition (changes in ownership)
    if "transfer" in lower_text or "ownership" in lower_text:
        return "E8 Acquisition"
    
    # E39 Actor (indigenous or cultural groups)
    if any(word in lower_text for word in ["ifugao", "igorot", "apayao", "kalinga"]):
        return "E39 Actor"
    
    # E21 Person (specific individuals)
    if lower_text in ["jose rizal", "apolinario mabini", "corazon aquino"]:
        return "E21 Person"
    
    # E40 Legal Body (formal organizations, government bodies)
    if "congress" in lower_text or "government" in lower_text or "legal" in lower_text:
        return "E40 Legal Body"
    
    # E53 Place (physical locations)
    if lower_text in ["maynila", "cebu", "batangas", "ermita", "luzon", "sagada", "tagaytay"]:
        return "E53 Place"
    
    # E44 Place Appellation (names or identifiers of places)
    if any(sym in entity_text for sym in ['"', "'"]):
        return "E44 Place Appellation"
    
    # E73 Information Object (textual/oral/digital representations)
    if "biag ni lam-ang" in lower_text or "oral tradition" in lower_text:
        return "E73 Information Object"
    
    # E41 Appellation (titles or labels)
    if "dance" in lower_text and "title" in lower_text:
        return "E41 Appellation"
    
    # E31 Document (written records/publications)
    if any(word in lower_text for word in ["manuskrito", "publication", "document", "pact"]):
        return "E31 Document"
    
    # E55 Type (classifications)
    if "type" in lower_text or "classification" in lower_text or "ritual" in lower_text:
        return "E55 Type"
    
    # E33 Linguistic Object (linguistic expressions)
    if "baybayin" in lower_text or "folk song" in lower_text:
        return "E33 Linguistic Object"
    
    # E84 Information Carrier (physical carriers, e.g., manuscripts, photographs)
    if any(word in lower_text for word in ["manuscript", "photograph", "carrier", "palm leaf"]):
        return "E84 Information Carrier"
    
    # Default fallback: use propositional object for abstract cultural ideas.
    return "E89 Propositional Object"
