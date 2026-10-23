from torch.utils.data import Dataset
from .prompts import BINARY_CLASSIFICATION_PROMPT


class BCDataset(Dataset):

    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        title = self.df.iloc[index]['TITLE']
        abstract = self.df.iloc[index]['ABSTRACT']
        prompt = BINARY_CLASSIFICATION_PROMPT % (title, abstract)
        return prompt
