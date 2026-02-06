# Animal-to-human (A2H) Translation (Preclinical Database and Obesity case study)

The preclinical database (PCDB) is an open-source database derived from the biomedical literature database, PubMed Central (PMC), and linked diseases, drug compounds, and animal species entities to public databases. Large language models were applied to filter publications searched through PMC via query templates with MeSH disease concepts and extracted entity mentions in the publications. All entites were linked and standardized to the public database based on their ontology structure to ensure proper semantic categories (e.g., disease entities are under MeSH Diseases category). As a case study, we connected preclinical evidence for Obesity drug development to corresponding clinical studies and identified key factors for lowering translation gaps.

This repository focuses on reproducible PCDB construction, including the entity linking and validation logics. It also provides analysis code for the A2H Obesity dataset for generating the insights presented in the research project publication.

![](./assets/A2H_PCDB_Obesity_overview.png)

## Authors
- Kaichi Xie — Graduate Student<sup>1,2,3</sup>
- Riyan Townsley — undergraduate Student<sup>1,3</sup>
- Ilias Tagkopoulos — Principal Investigator<sup>1,2,3</sup>

<sup>1</sup> Department of Computer Science, University of California at Davis
<sup>2</sup> Genome Center, University of California at Davis
<sup>3</sup> USDA/NSF AI Institute for Next Generation Food Systems (AIFS)

## Contact
Questions, bug reports, and collaboration ideas are welcome at Kaichi Xie (`kcxie@ucdavis.edu`) or Prof. Ilias Tagkopoulos (`itagkopoulos@ucdavis.edu`).

## Citation
If you use A2H or PCDB in your research, please cite.

## License
 released under the Apache-2.0 License. See [`LICENSE`](./LICENSE) for the full text and usage terms.


## Funding
This work is supported by the USDA-NIFA AI Institute for Next Generation Food Systems (AIFS), USDA-NIFA award number 2020-67021-32855.
