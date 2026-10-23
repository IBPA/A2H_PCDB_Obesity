BINARY_CLASSIFICATION_PROMPT= """
Title:
%s

Abstract:
%s

Question:
Given the title and abstract of a research article above, does this article test a drug's potential as a treatment for a clearly defined human disease through in vivo experiments conducted on an animal model?
The study must clearly state the disease being treated and the drug being tested, and focus on evaluating the therapeutic efficacy of the drug in a disease model and an animal model.

Definition and exclusions:
Drug should be a substance recognized by an official pharmacopeia or formulary, intended for use in the diagnosis, cure, mitigation, treatment, or prevention of disease, and affects the structure or any function of the body. It is not merely a component or accessory of a device. Biological products are included within this definition and are generally covered by the same laws and regulations, but differences exist regarding their manufacturing processes (chemical process versus biological process.)
Research focusing on drug delivery systems (e.g., nanoparticles) without specifying a drug does not qualified.
Research focusing on drug pharmacokinetics, drug distribution, or drug toxicity does not qualified.
Herbs, foods, dietary supplements, plants, or plant extracts do not qualify as drugs but specific chemical compounds or molecules from these materials qualify as drugs.

Answer the question with yes or no only.

Answer:
"""
