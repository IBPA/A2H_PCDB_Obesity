"""Entity-extraction prompts (Stage 3), as reported in the Supplementary Information.

Used by extract_entities.py, which also sets the decoding parameters for each step.
Templates use Python %-formatting, hence the '%%' escapes in the example text.
"""

SYSTEM_PROMPT = "You are an expert in medicine and drug discovery."

DISEASE_QUESTION = (
    "The article describes in-vivo preclinical experiments involving the testing of drugs on animal models "
    "to treat human diseases. Extract the full names of the primary human diseases that the drug in the article "
    "directly targets for treatment. Provide them in the specified JSON format. If no human disease is explicitly "
    "targeted, leave the JSON array empty for the corresponding field."
)

# Input: title + abstract. One-shot example: Salloum et al., Circulation 120 (2009).
DISEASE_EXTRACTION_PROMPT = """Title: Phosphodiesterase-5 Inhibitor, Tadalafil, Protects against Myocardial Ischemia/Reperfusion through Protein-Kinase G Dependent Generation of Hydrogen Sulfide

Abstract: Tadalafil is a novel long acting inhibitor of phosphodiesterase-5. Since cGMP-dependent protein kinase (PKG) signaling plays a key role in cardioprotection, we hypothesized that PKG activation with tadalafil would limit myocardial ischemia/reperfusion (I/R) injury and dysfunction. Additionally, we contemplated that cardioprotection with tadalafil is mediated by hydrogen sulfide (H2S) signaling in a PKG-dependent fashion. After baseline transthoracic echocardiography (TTE), adult ICR mice were injected i.p. with vehicle (10%% DMSO) or tadalafil (1 mg/kg) with or without KT5823 (KT, PKG blocker, 1 mg/kg) or dl-propargylglycine [PAG, Cystathionine-γ-lyase (CSE, H2S-producing enzyme) blocker; 50 mg/kg] 1 h prior to coronary artery ligation for 30 min and reperfusion for 24 h, whereas C57BL-wild type and CSE-knockout mice were treated with either vehicle or tadalafil. After reperfusion, TTE was performed and hearts were collected for infarct size (IS) measurement using TTC staining. Survival was increased with tadalafil (95%%) compared with control (65%%, P<0.05). Infarct size was reduced with tadalafil (13.2±1.7%%) compared to vehicle (40.6±2.5%%; P<0.05). PKG activation with tadalafil limits myocardial infarction and preserves LV function through H2S signaling.

Question: """ + DISEASE_QUESTION + """
Answer:
{
    "diseases": ["myocardial ischemia/reperfusion injury"]
}

Title: %s
Abstract: %s
Question: """ + DISEASE_QUESTION + """
Answer:
"""

# Drug extraction, step 1 of 2. Input: Results section.
DRUG_SUMMARY_PROMPT = """Results: %s
Instruction:
The above text is the results section of an article that includes in-vivo experiments for testing drugs on animals to treat human diseases.
Generate a detailed summary for all in-vivo experiments for drug testing.
Only include the drug treatment groups tested on animals and in-vivo, and exclude any in-vitro experiments.
Include drug names exactly as they appear in the paper, without any modifications or assumptions.
Include the full names of the drugs when available and abbreviations when only those are provided.
Summary:
"""

# Drug extraction, step 2 of 2. Input: the summary generated in step 1.
DRUG_EXTRACTION_PROMPT = """Summary: %s
Instruction:
The above text is a summary of in-vivo experiments for testing drugs on animals to treat human diseases.
Extract the full names of the drugs tested in vivo as treatments for human diseases, including the positive control.
You must include drug names exactly as they appear in the paper, without any modifications or assumptions.

Focus on extracting the names of treatments explicitly tested for human diseases, excluding those used solely as disease-inducing agents.
Include only treatments intended for therapeutic use and exclude those primarily used to study biological mechanisms.
Exclude dosage information.
Provide them in the specified JSON format and use " + " for tested drug combinations: {
    "drugs": ["drug_a", "drug_b", "drug_c + drug_d"]
}.
Provide the JSON as the output only.
Answer:
"""

ANIMAL_QUESTION = """Question: The text is from an article describing in-vivo experiments for testing drugs on animal models for the treatment of human diseases. What animals are tested in vivo in the article?
Based on the provided text, extract the common names of strain and species and total subject size of each animal model that are only used in drug testing experiment.
Provide them in a JSON format only. Provide only the JSON as the output. Do not include explanations or text outside of the JSON. If no information is found, leave the corresponding field empty:
Answer:"""

# Input: Methods section. Only species is used downstream; strain and subject size are kept as attributes.
ANIMAL_EXTRACTION_PROMPT = """Method: "Adult male out-bred ICR mice (n = 10) were supplied by Harlan Sprague Dawley (Indianapolis, IN). The mean body weight was 32.2 ± 0.4 g. CSE-knockout mice (4 males and 5 females) were kindly provided by Dr. David J. Lefer (Emory University School of Medicine)."
""" + ANIMAL_QUESTION + """
[{
    "strain": "ICR",
    "species": "mice",
    "total subject size": "10"
},
{
    "strain": "CSE-knockout",
    "species": "mice",
    "total subject size": "9"
}]

Method: %s
""" + ANIMAL_QUESTION + "\n"
