import pandas as pd
import numpy as np
import json
import gzip
import sys
import os
from Bio import SeqIO
#import swifter

#update these lines as needed
api_dir = './'
ps_data_dir = './ProteomeScoutAPI/proteomescout_mammalia_20220131/data.tsv'
source_data_dir = './ensembl_data/'
processed_data_dir = './processed_data_dir/'
translator_file = 'uniprot_translator.csv'
available_transcripts_file = processed_data_dir + 'available_transcripts.json'

if os.path.isfile(available_transcripts_file):
    with open(available_transcripts_file, 'r') as f:
        available_transcripts = json.load(f)
else:
    print('Indicated available transcript file does not exist. Run.... \n')
    available_transcripts = None

#load ProteomeScoutAPI
sys.path.append(api_dir)
from ProteomeScoutAPI import proteomeScoutAPI
ps_api = proteomeScoutAPI.ProteomeScoutAPI(ps_data_dir)

### Utility Functions ###
def getCanonicalIDs(translator, ID_type = 'Transcript'):
    """
    Use translator to extract only the canonical proteins (either Uniprot IDs or Transcript IDs)
    """
    if ID_type == 'Transcript':
        return translator[translator['canonicals']=='canonical']['Transcript stable ID'].to_list()
    elif ID_type == 'Protein':
        return translator[translator['canonicals']=='canonical']['Uniprot ID'].to_list()
    else:
        print("Please indicate whether you want 'Transcript' or 'Protein' IDs")
        return None


### Processing Functions ###
def processTranslator(ensembl_translator_dir):
    #load info to translate between proteomeScout
    translator = pd.read_csv(ensembl_translator_dir, compression = 'gzip')
    translator.dropna(inplace = True)
    #translator= pd.concat([pd.Series(row['Entry'], row['Ensembl transcript'].split(';'))			  
    #                    for _, row in translator.iterrows()]).reset_index()
    #translator.columns = ['Transcript and isoform', 'Uniprot ID']
    #get the transcript ID alone
    #translator['Transcript stable ID'] = translator.apply(just_transcript, axis = 1)
    #get the uniprot ID alone
    #translator['Isoform'] = translator.apply(just_isoform, axis = 1)
    #remove any empty rows
    #translator.dropna(inplace = True)
    #indicate whether the transcript/protein is the canonical protein
    translator['canonicals'] = translator.apply(is_canonical, axis = 1)
    return translator


def just_transcript(row):
    """
    From ensemble translation data file, isolate just the transcript ID
    """
    entry= row['Transcript and isoform']
    if entry == '':
        transcript = 'N/A'
    elif len(entry.split()) == 1:
        transcript = entry
    else:
        transcript = entry.split()[0]
        
    return transcript
    
def just_isoform(row):
    """
    From ensemble translation data file, isolate just the Uniprot ID
    """
    entry= row['Transcript and isoform']
    if entry == '':
        isoform = np.nan
    elif len(entry.split()) == 1:
        isoform = '1'
    else:
        isoform = entry.split()[1]
        
    return isoform
    
def is_canonical(row):
    """
    Based on the uniprot ID, label each protein as canonical or isoform
    """
    if row['UniProtKB/Swiss-Prot ID'] is np.nan:
        ans = np.nan
    elif row['UniProtKB isoform ID'] is np.nan:
        ans = 'Canonical'
    elif row['UniProtKB isoform ID'].split('-')[1] == 1:
        ans = 'Canonical'
    else:
        ans = 'Alternative'
        
    return ans
    
    
def processEnsemblFasta(file, id_col = 'ID', seq_col = 'Seq'):
    data_dict = {id_col:[],seq_col:[]}
    with gzip.open(file,'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            ids = record.id
            seq = str(record.seq)
            
            data_dict[id_col].append(ids)
            data_dict[seq_col].append(seq)
            
    return pd.DataFrame(data_dict)
    
def perfect_align(row):
	
	u = row['PS Seq']
	v = row['GENCODE Seq']
	ans = u==v
	
	return ans


#load uniprot translator dataframe, process if need be
if os.path.isfile(processed_data_dir + translator_file):
    translator = pd.read_csv(processed_data_dir + translator_file)
elif os.path.isfile(source_data_dir + translator_file+'.gz'):
    translator = pd.read_csv(source_data_dir + translator_file+'.gz')
    if 'Uniprot Isoform' not in translator.columns:
        translator['Uniprot Canonical'] = translator.apply(is_canonical, axis =1)
    translator.to_csv(processed_data_dir + translator_file)
else:
    print('Provided translator file does not exist in source data directory or in processed data directory. Update config file.')






