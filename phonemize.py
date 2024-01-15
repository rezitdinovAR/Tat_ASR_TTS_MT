import pandas as pd
from text.cleaners import tatar_cleaners
from tqdm import tqdm
import sys

i=158

data = pd.read_csv(f"filelists/multispeaker.csv",sep='\t')
data = data.drop(['Unnamed: 0'],axis=1)

if i==158:
    for j in tqdm(range(i*1500, len(data)-1)):
        data.at[j,'phonemes'] = tatar_cleaners(data['text'][j])
    print('Обработали 1500')
    
    data.reset_index(drop=True).to_csv(f"filelists/multispeaker.csv", sep="\t")
    del data

else:
    for j in tqdm(range(i*1500, (i+1)*1500)):
        data.at[j,'phonemes'] = tatar_cleaners(data['text'][j])
    print('Обработали 1500')
    
    data.reset_index(drop=True).to_csv(f"filelists/multispeaker.csv", sep="\t")
    del data
