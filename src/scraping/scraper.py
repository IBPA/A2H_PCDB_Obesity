from lxml import etree 
import requests 
import tarfile 
import os     
import re
import aiofiles
import aioftp


class PMCArticleScraper: 

    def __init__(self):
        """
        Initializes PMCArticleScraper with some base_urls and namespaces for text extraction
        
        """
        self.base_url = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/" #OAI-PMH API 
        self.ns = {
            "jats": "https://jats.nlm.nih.gov/ns/archiving/1.4/",
        }
        
    def _fetch_article_xml(self, pmc_id: int) -> bytes: 
        """
        Fetches raw XML for article 
    
        Args: 
            pmc_id (int): ID of PMC article  
            
        Returns: 
            bytes: raw bytes of XML object    
        """
        
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:pubmedcentral.nih.gov:{pmc_id}", 
            "metadataPrefix": "pmc"  
        }
        
        response = requests.get(self.base_url, params=params)
        
        return response.content
    
    def _save_text_to_file(self, file_path: str, text: str) -> None: 
        """
        Writes input text to file path of choice 
        
        Args: 
            file_path (str): file path to write to 
            text (str): text to write 
           
        """
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
            

    def _retrieve_text(self) -> str: 
        """
        Cleans and Extracts Full-Article Text from PMC Article
        
        Args: 
            pmc_id (int): PMC ID of article to Scrape
            
        Return: 
            str: Article Text (excluding references)
            
        """
        xml_data = self._fetch_article_xml() 

        root = etree.fromstring(xml_data)

        sections = root.findall('.//jats:sec', namespaces=self.ns)
        
        cleaned_texts = []

        for sec in sections:
            title = sec.find('jats:title', namespaces=self.ns)
            
            if title is not None and title.text:
                cleaned_texts.append(title.text.strip())

            paragraphs = sec.findall('jats:p', namespaces=self.ns)
            
            #Cleans and rewrites references into readable format
            for p in paragraphs:
                
                if p.get('content-type') == 'self-citation':
                    continue  

                for xref in p.findall('.//jats:xref', namespaces=self.ns):
                    
                    if xref.get('ref-type') == 'bibr':
                        sup = xref.find('.//jats:sup', namespaces=self.ns)
                        
                        if sup is not None and sup.text:
                            
                            parent = xref.getparent()
                            idx = parent.index(xref)
                            new_tail = (xref.tail or '')
                            parent.remove(xref)
                            new_span = etree.Element("span")
                            new_span.text = f"({sup.text})" + new_tail
                            parent.insert(idx, new_span)

                paragraph_text = ''.join(p.itertext()).replace('\n', ' ').strip()
                cleaned_texts.append(paragraph_text)
                
        full_text = '\n\n'.join(cleaned_texts)
          
        return full_text


                    
    async def download_pmc_ftp(self, tgz_url: str, save_dir: str) -> None:
        """
        Downloads a .tar.gz file through FTP 
        
        Args: 
            tgz_url (str): Extracted .tar.gz url (from PMC OA) 
            save_dir (str): Directory being saved to 
            
        Return: 
            None 
            
        """
        ftp_host = "ftp.ncbi.nlm.nih.gov"
        ftp_path = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "")

        async with aioftp.Client.context(ftp_host) as client:
            async with client.download_stream(ftp_path) as stream:
                async with aiofiles.open(save_dir, "wb") as f:
                    async for block in stream.iter_by_block():
                        await f.write(block)
                        
        print(f"Finished downloading to {save_dir}")

    async def fetch_and_extract(self, pmc_id: int, out_dir: str) -> None:
        """
        For given PMC ID, extract and decompresses its tar.gz file through FTP and saves to provided directory.
        
        Args: 
            pmc_id (int): ID of PMC Article 
            out_dir (str): directory that extracted tar.gz contents are written to 
        
        Return: 
            None 

        """
        article_id = 'PMC' + str(pmc_id)
        api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={article_id}"
        response = requests.get(api_url)
        
        root = etree.fromstring(response.text)

        tgz_url = root.find(".//link[@format='tgz']").attrib['href']
        save_path = f"PMC{pmc_id}.tar.gz" #save_path to temporarily save to 
        
        os.makedirs(out_dir, exist_ok=True)

        await self.download_pmc_ftp(tgz_url, save_path)
    
        with tarfile.open(save_path, 'r:gz') as tar:
            tar.extractall(path=out_dir, filter="data")
     
        
        os.remove(save_path)
        
       

    def _clean_caption_text(self, text: str) -> str:
        """
        Cleans figure caption text by reformatting.
        
        Args: 
            text (str): text to reformat 
            
        Return: 
            str: Cleaned text 
        """
        text = text.replace('\xa0', ' ')          
        text = re.sub(r'\s*=\s*', ' = ', text)        
        text = re.sub(r'\s+', ' ', text)            
        text = text.strip()    
                            
        return text

    def _retrieve_figure_captions(self, pmc_id: int) -> list:
        """
        Extracts and cleans figure captions from PMC XML.

        Args: 
            pmc_id (int): ID of Article whose figure captions will be extracted
            
        Return: 
            list: list of str of the article captions 
            
        """
        xml_data = self._fetch_article_xml(pmc_id)
        root = etree.fromstring(xml_data)

        captions = []
        figures = root.findall(".//jats:fig", namespaces=self.ns)

        for fig in figures:
            caption_elem = fig.find(".//jats:caption", namespaces=self.ns)
            
            if caption_elem is not None:
                paras = caption_elem.findall("jats:p", namespaces=self.ns)
                raw_caption = " ".join(" ".join(p.itertext()).strip() for p in paras)
                cleaned = self._clean_caption_text(raw_caption)
                
                if cleaned:
                    captions.append(cleaned)

        return captions
    
    
    def extract_fig_number(self, label: str): 
        if not label: 
            return None 
        
        label = label.strip() 
        
        number_match = re.search(r'(\d{1,2})(?!\d)', label)
        
        if number_match: 
            return int(number_match.group(1))
        
        return None 
        

        
        pass 
    
    def _retrieve_fig_caption_map(self, root_dir: str, id_list: list) : 
        """
        
        """
        fig_to_caption_dict = {} 
        
        for pmc_id in id_list: 
            
            file_path_prefix = root_dir + "/" + "PMC" + str(pmc_id)
            
            
            xml_data = self._fetch_article_xml(pmc_id)

            root = etree.fromstring(xml_data)
         
            figures = root.findall(".//jats:fig", namespaces=self.ns)
            
            for fig in figures: 
                
                fig_label  = fig.find(".//jats:label", namespaces=self.ns)
                
                if fig_label is None: 
                    continue
                
                figure_id = fig_label.text
                
                
                graphic_info  = fig.find(".//jats:graphic", namespaces=self.ns)
                href = graphic_info.get('{http://www.w3.org/1999/xlink}href')

                
                if figure_id and href: 
                    figure_number = self.extract_fig_number(figure_id)
                    
                    
                    figure_name = href 
                    
                    file_path = file_path_prefix + "/" + figure_name
                    if os.path.isfile(file_path_prefix + "/" +  "Figure_" + str(figure_number) + "_Caption.txt"): 
                        
                        fig_to_caption_dict[file_path] = file_path_prefix + "/" +  "Figure_" + str(figure_number) + "_Caption.txt"
                        
             
           
        print("Finished Mapping Captions to Figures")
          
        return fig_to_caption_dict   
        
    
#For Testing
if __name__ == "__main__":
    rand_object = PMCArticleScraper()
    print(rand_object._fetch_article_xml(9796023))

