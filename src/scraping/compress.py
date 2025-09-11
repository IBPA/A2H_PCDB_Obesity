import pandas as pd 
import os 
import shutil
import sys 

def main(root_dir: str, filtered_csv: str, zip_fname: str): 
    
    df = pd.read_csv(filtered_csv, engine="python", on_bad_lines="skip")
    filtered_df = df[df["classification_result"] == "yes"]

    os.makedirs(root_dir,exist_ok=True)
        
    for _, row in filtered_df.iterrows(): 
        figure_path = row["figure_path"]
        
        parts = figure_path.split(os.sep)

        subfolder_path = root_dir + "/" + parts[1]
        os.makedirs(subfolder_path, exist_ok=True)
        
        dest_path = os.path.join(subfolder_path, os.path.basename(figure_path))
        shutil.copy(figure_path, dest_path)
    
    shutil.make_archive(zip_fname, "zip", root_dir)
    shutil.rmtree(root_dir)
    
    
if __name__ == "__main__": 
    if len(sys.argv) < 3:
        print("Usage: python3 compress.py dir_to_create filtered_csv_name zip_file_name")
        sys.exit(1)

    new_dir_name = sys.argv[1] 
    csv_name = sys.argv[2] 
    zip_file_name = sys.argv[3]
    
    
    main(new_dir_name, csv_name, zip_file_name)

