import pandas as pd
from text.cleaners import tatar_cleaners
from tqdm import tqdm
import sys

data = pd.read_csv("filelists/AA44_base.csv",sep='\t')
data = data.drop(['Unnamed: 0'],axis=1)

i = int(sys.argv[1])

if i==10:
    for j in tqdm(range(i*1500, len(data)-1)):
        data.at[j,'phonemes'] = tatar_cleaners(data['text'][j])
    print('Обработали all')
    
    data.reset_index(drop=True).to_csv(f"filelists/AA44_base.csv", sep="\t")
    del data

else:
    for j in tqdm(range(i*1500, (i+1)*1500)):
        data.at[j,'phonemes'] = tatar_cleaners(data['text'][j])
    print(f"Обработали {(i+1)*1500}")
    
    data.reset_index(drop=True).to_csv(f"filelists/AA44_base.csv", sep="\t")
    del data
