from openai import OpenAI
from scraper import PMCArticleScraper
import base64, mimetypes, pathlib
import pandas as pd
import os 
import sys 

def to_data_url(path: str) -> str:
    path = pathlib.Path(path).expanduser()
    mime, _ = mimetypes.guess_type(path)      
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

def get_figure_caption(caption_file_path: str): 
    if caption_file_path is None: 
        return None 
    
    with open(caption_file_path, 'r') as file: 
        caption = file.read() 
        return caption


def main(file_dir: str, target_value: str, write_file_name: str): 
    
    client = OpenAI()
    
    scraper = PMCArticleScraper()

    files = []
    article_ids = []
    
    for folder in os.listdir(file_dir): 
        folder_path = os.path.join(file_dir, folder)
        article_ids.append(folder[3:])
        
        
        if os.path.isdir(folder_path): 
            for file in os.listdir(folder_path): 
                if file.endswith('.jpg'): 
                    file_path = os.path.join(folder_path, file)
                    files.append(file_path)
    
    
    fig_to_caption_dict = scraper._retrieve_fig_caption_map(file_dir, article_ids)
    
    
    figure_captions = []
    classification_results = []
    
    for figure_path in files: 
        encoded_img_url = to_data_url(figure_path)
        figure_caption = get_figure_caption(fig_to_caption_dict.get(figure_path, None))

        figure_captions.append(str(figure_caption))
        
        prompt = f"""
        Determine if the target outcome value is explicitly measured in the provided scientific graph plot.

        Target outcome value: "{target_value}"

        Focus only on the y-axis text and any relevant visual cues. Ignore all other considerations unless they help confirm whether the y-axis or measurement aligns with the target value.

        .output_text: respond with exactly one word, either "yes" or "no", all lowercase, with no explanation or additional text.

        Do not include reasoning. Do not say anything else. Only respond with "yes" or "no".
        """
        
        prompt_with_caption = f"""
        Determine if the target outcome value is explicitly measured in the provided scientific graph plot.

        Target outcome value: "{target_value}"

        Focus only on the y-axis text and any relevant visual cues. Ignore all other considerations unless they help confirm whether the y-axis or measurement aligns with the target value.

        .output_text: respond with exactly one word, either "yes" or "no", all lowercase, with no explanation or additional text.

        Do not include reasoning. Do not say anything else. Only respond with "yes" or "no".
        
        To provide additional context: here's the caption for the figure - {figure_caption}
        """
        
        if figure_caption== None: 
            response = client.responses.create(
                        # model = "o3",
                        model="gpt-4.1",     
                       
                        input = [{
                            "role": "user", 
                            "content": [ 
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", 
                                "image_url": encoded_img_url,},      
                            ],    
                        }],
                    
                    )
            
            
        else: 
            response = client.responses.create(
                        # model = "o3",
                        model="gpt-4.1",     
            
                        input = [{
                            "role": "user", 
                            "content": [ 
                                {"type": "input_text", "text": prompt_with_caption},
                                {"type": "input_image", 
                                "image_url": encoded_img_url,},      
                            ],    
                        }],
                    
                    )
            
        classification_result = response.output_text
        classification_results.append(classification_result)
    
    data = {
        'figure_path': files, 
        'caption': figure_captions, 
        'classification_result': classification_results
    } 
    
    
    df = pd.DataFrame(data) 
    
    df.to_csv(write_file_name,index=False)

if __name__ == '__main__': 
    main("test", "Body Weight")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 filter.py path/to/pmc_dir outcome_measure_str output_csv_name")
        sys.exit(1)

    pmc_file_dir = sys.argv[1] #Root directory of PMC articles donwloaded from previous step
    target_metric = sys.argv[2] #Outcome measure you want to filter by (e.g body weight for obesity)
    output_csv = sys.argv[3] #CSV file name to write to 
    
    main(pmc_file_dir, target_metric, output_csv)



