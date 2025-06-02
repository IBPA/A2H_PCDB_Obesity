import asyncio
import os
import sys
import random
import pandas as pd
from scraper import PMCArticleScraper
import aiofiles
from asyncio.exceptions import TimeoutError



SEM_LIMIT = 5 # Max number of concurrent async tasks
MAX_RETRIES = 3 # Retry limit for download/timeout errors 
TIMEOUT_SECONDS = 30 #Timeout Limit for task



async def process_article(pmc_id: str, out_dir: str, semaphore: asyncio.Semaphore) -> None:
    """
    Asynchronously extracts articles with a semaphore to limit concurrent tasks
    """
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                scraper = PMCArticleScraper()
                await asyncio.wait_for(scraper.fetch_and_extract(pmc_id, out_dir), timeout=TIMEOUT_SECONDS)

                captions = scraper._retrieve_figure_captions(pmc_id)
                figure_dir = os.path.join(out_dir, f"PMC{pmc_id}")
                os.makedirs(figure_dir, exist_ok=True)

                for i, caption in enumerate(captions):
                    
                    file_path = os.path.join(figure_dir, f"Figure_{i+1}_Caption.txt")
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                        await f.write(caption)

                print(f"Finished PMC ID {pmc_id}")
                return

            except TimeoutError:
                print(f"Timeout: PMC ID {pmc_id} (Attempt {attempt}/{MAX_RETRIES})")

            except AttributeError as e:
                if "'NoneType' object has no attribute 'attrib'" in str(e):
                    print(f"Skipping PMC ID {pmc_id} as it's not available for download")
                    return 
                
            except Exception as e:
                print(f"Failed PMC ID {pmc_id} (Attempt {attempt}/{MAX_RETRIES}): {e}")

            if attempt < MAX_RETRIES:
                cooldown_time = (2 ** (attempt - 1)) + random.gauss(0.5, 0.1) #Exponential backoff + jitter
                await asyncio.sleep(cooldown_time)
                
            else:
                print(f"Failed to extract PMC Article {pmc_id} after {MAX_RETRIES} attempts")

def load_pmc_ids(csv_path: str) -> list:
    """
    Loads PMC ID from CSV file 

    Args: 
        csv_path (str): path to csv file 
        
    Returns: 
        list: list of unique PMC IDs
    """
    #df = pd.read_csv(csv_path, sep="\t", engine="python", on_bad_lines='skip')
    # Uncomment and run above line instead if .tsv file 
    
    df = pd.read_csv(csv_path)
    
    if "pmcid" not in df.columns:
        raise ValueError("CSV must contain a 'pmcid' column.")
    
    return list(set(int(pmcid[3:]) for pmcid in df["pmcid"]))

async def main(csv_path: str, out_dir: str) -> None:
    """
    Main function for async article extraction 
    
    """
    pmc_ids = load_pmc_ids(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(SEM_LIMIT)
    
    tasks = [process_article(pmc_id, out_dir, semaphore) for pmc_id in pmc_ids]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scrape_figures.py path/to/articles.csv output_dir/")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    asyncio.run(main(csv_path, out_dir))
