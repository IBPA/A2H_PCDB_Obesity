"""Write a labeled TSV as the question/answer JSON that torchtune's instruct_dataset reads."""
import json

from .prompts import BINARY_CLASSIFICATION_PROMPT


def create_json_file(dataframe, save_path):
    data_points = []
    for _, row in dataframe.iterrows():
        prompt = BINARY_CLASSIFICATION_PROMPT % (row['TITLE'], row['ABSTRACT'])
        data_points.append({"question": prompt,
                            "answer": row['label']})

    with open(save_path, 'w') as json_file:
        json.dump(data_points, json_file, indent=4)

    print(f"JSON file has been created at {save_path}")
