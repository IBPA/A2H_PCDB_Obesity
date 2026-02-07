import pandas as pd
import ast
import pickle
import inflect

from data.PCDB.external_database_mapping_utils.animal_normalization_maps import (
    ANIMAL_SYNONYM_MAP,
    MACAQUE_SPECIES_MAP,
    MICE_RATS_STRAIN_CLEANUP_MAP
)


# Initialize inflect engine for singular/plural conversion
p = inflect.engine()


def load_mesh_animal_data():
    """Load MeSH animal lookup tables from pickle files."""
    with open("data/PCDB/external_database_mapping_utils/external_database_lut/mesh_animal_ids_LUT.pkl",
              "rb") as f:
        animal_mesh_id_LUT = pickle.load(f)

    with open("data/PCDB/external_database_mapping_utils/external_database_lut/mesh_animal_descriptors_LUT.pkl",
              "rb") as f:
        animal_mesh_descriptor_LUT = pickle.load(f)

    return animal_mesh_id_LUT, animal_mesh_descriptor_LUT


def normalize_animal(raw):
    """Normalize animal species name using synonym dictionary."""
    if raw is None:
        return None

    s = raw.strip().lower()
    if not s:
        return None

    if s in ANIMAL_SYNONYM_MAP:
        return ANIMAL_SYNONYM_MAP[s]
    else:
        print(f"{s} not resolved yet.")
        return ""


def normalize_monkey_label(strain, species_original):
    """
    Normalize monkey/macaque species based on strain information.
    This part can be ignored in the future.

    Input:  e.g. 'cynomolgus - monkey'
    Output: e.g. 'cynomolgus macaque'
    For unrecognized / missing strains => 'macaque'
    """
    species = MACAQUE_SPECIES_MAP.get(strain)

    if species is not None:
        return species
    else:
        # print(f"Unable to normalize: {strain} - {species_original} ---> macaque")
        return 'macaque'


def animal_species_map_to_mesh(species, animal_mesh_id_LUT):
    """Map animal species to MeSH ID."""
    animal_species = species
    if animal_species == "":
        return ""
    if animal_species in animal_mesh_id_LUT:
        return {"mesh": animal_mesh_id_LUT[animal_species][0]}

    animal_species_singular = p.singular_noun(animal_species)
    if animal_species_singular and animal_species_singular in animal_mesh_id_LUT:
        return {"mesh": animal_mesh_id_LUT[animal_species_singular][0]}

    animal_species_plural = p.plural_noun(animal_species)
    if animal_species_plural and animal_species_plural in animal_mesh_id_LUT:
        return {"mesh": animal_mesh_id_LUT[animal_species_plural][0]}

    return ""


def animal_map_to_mesh(row, animal_mesh_id_LUT):
    """Map animals in a row to MeSH IDs."""
    animal_data = row["animals"]

    mapped_animals = []
    unmapped_animals = []

    if animal_data != "" and len(animal_data) > 0:
        for animal in animal_data:
            if animal == '':
                continue
            strain = animal['strain'].lower()
            species = animal['species'].lower()

            if ("animal" in species) or ("human" in species):
                continue

            total_subject_size = animal["total subject size"]

            mesh_mapping = animal_species_map_to_mesh(species, animal_mesh_id_LUT)

            if mesh_mapping != "":
                species = normalize_animal(species)
                if species == "monkey" or species == "macaque":
                    species = normalize_monkey_label(strain, species)
                    strain = ''
                    mesh_mapping = animal_species_map_to_mesh(species, animal_mesh_id_LUT)
                    if mesh_mapping == "":
                        print(species)

                if mesh_mapping and mesh_mapping['mesh'] in MICE_RATS_STRAIN_CLEANUP_MAP:
                    if mesh_mapping['mesh'] != 'D008819':
                        strain = MICE_RATS_STRAIN_CLEANUP_MAP[mesh_mapping['mesh']]['strain']
                    species = MICE_RATS_STRAIN_CLEANUP_MAP[mesh_mapping['mesh']]['species']
                    mesh_mapping['mesh'] = MICE_RATS_STRAIN_CLEANUP_MAP[mesh_mapping['mesh']]['mesh']
                    # print(f"Cleaned mice/rat strain: {strain}, {species}, {mesh_mapping['mesh']}")

                if species == '':
                    unmapped_animals.append({
                        "strain": "",
                        "species": "",
                        "total_subject_size": ""
                    })
                else:
                    mapped_animals.append({
                        "strain": strain,
                        "species": species,
                        "total_subject_size": total_subject_size,
                        "mesh": mesh_mapping["mesh"]
                    })
            else:
                unmapped_animals.append({
                    "strain": strain,
                    "species": species,
                    "total_subject_size": total_subject_size
                })

    if len(mapped_animals) == 0:
        mapped_animals = ""
    if len(unmapped_animals) == 0:
        unmapped_animals = ""

    return mapped_animals, unmapped_animals


def main():
    # Load PCDB database
    print("Loading PCDB database...")
    pcdb = pd.read_csv(
        "data/PCDB/preclinical_database_raw.tsv",
        sep="\t", keep_default_na=False, dtype=str
    )

    # Parse animals column
    pcdb["animals"] = pcdb["animals"].apply(lambda x: ast.literal_eval(x) if x != '' else x)
    print(f"Total rows: {len(pcdb)}")

    # Load MeSH animal data and build lookup tables
    print("\nLoading MeSH animal data...")
    animal_mesh_id_LUT, animal_mesh_descriptor_LUT = load_mesh_animal_data()
    print(f"Total animal MeSH terms: {len(animal_mesh_id_LUT)}")

    # Map animals to MeSH IDs
    print("\nMapping animals to MeSH IDs...")
    pcdb[["mapped_animals", "unmapped_animals"]] = pcdb.apply(
        lambda x: animal_map_to_mesh(x, animal_mesh_id_LUT),
        axis=1, result_type="expand"
    )

    # Calculate statistics
    mapped_count = len(pcdb[pcdb["mapped_animals"] != ""])
    unmapped_count = len(pcdb[pcdb["unmapped_animals"] != ""])
    print(f"\nArticles with mapped animals: {mapped_count} ({mapped_count/len(pcdb)*100:.2f}%)")
    print(f"Articles with unmapped animals: {unmapped_count}")

    # Save results
    print("\nSaving results...")
    pcdb.to_csv("outputs/pcdb/mapped_animals.tsv", sep="\t", index=False)
    print("Results saved to: outputs/pcdb/mapped_animals.tsv")

    print(pcdb.head())


if __name__ == "__main__":
    main()
