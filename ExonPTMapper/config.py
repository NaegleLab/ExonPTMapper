import pandas as pd
import numpy as np
import json
import gzip
import sys
import os
import pybiomart
from Bio import SeqIO
#import swifter

#update these lines as needed
api_dir = 'C:\\Users\Sam\OneDrive\Documents\GradSchool\Research'
ps_data_dir = 'C:\\Users\Sam\OneDrive\Documents\GradSchool\Research\ProteomeScoutAPI\proteomescout_mammalia_20220131\data.tsv'
source_data_dir = 'C://Users/Sam/OneDrive/Documents/GradSchool/Research/Splicing/Data_April252023/ensembl_data/'
processed_data_dir = 'C://Users/Sam/OneDrive/Documents/GradSchool/Research/Splicing/Data_April252023/processed_data_dir/'
#source_data_dir = '../Ensembl54_Data/ensembl_data/'
#processed_data_dir = '../Ensembl54_Data/processed_data_dir/'
available_transcripts_file = processed_data_dir + 'available_transcripts.json'



def is_canonical(row):
    """
    Based on the uniprot ID, label each protein as canonical or isoform
    """
    if row['UniProtKB/Swiss-Prot ID'] is np.nan:
        ans = np.nan
    elif row['UniProtKB isoform ID'] is np.nan:
        ans = 'Canonical'
    elif int(row['UniProtKB isoform ID'].split('-')[1]) == 1:
        ans = 'Canonical'
    else:
        ans = 'Alternative'
        
    return ans
    


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


#load uniprot translator dataframe, process if need be
print('Downloading ID translator file')
if os.path.isfile(processed_data_dir + 'translator.csv'):
    translator = pd.read_csv(processed_data_dir + 'translator.csv')
else:
    chromosomes = ['X', '20', '1', '6', '3', '7', '12', '11', '4', '17', '2', '16',
       '8', '19', '9', '13', '14', '5', '22', '10', 'Y', '18', '15', '21',
       'MT']
    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',
                   host='http://www.ensembl.org')
    translator = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id',
                                       'uniprotswissprot', 'uniprot_isoform'],
             filters = {'biotype':'protein_coding','transcript_biotype':'protein_coding',
             'chromosome_name':chromosomes})
    translator2 = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id', 'pdb','refseq_mrna', 'ccds'],
             filters = {'biotype':'protein_coding','transcript_biotype':'protein_coding',
             'chromosome_name':chromosomes})
    translator = translator.merge(translator2, on = ['Gene stable ID', 'Gene name', 'Transcript stable ID'], how = 'outer')
    translator = translator.drop_duplicates()
    #indicate whether row contains information on uniprot canonical transcript
    translator['Uniprot Canonical'] = translator.apply(is_canonical, axis =1)
    
    translator.to_csv(processed_data_dir + 'translator.csv')
    
if os.path.isfile(processed_data_dir + 'isoforms.csv'):
    isoforms = pd.read_csv(processed_data_dir + 'isoforms.csv')


def getIsoformInfo(transcripts):
    """
    Get dataframe where each row corresponds to a unique protein isoform, with transcripts with identitical protein information being in the same row
    """
    #add amino acid sequence to translator information
    merged = config.translator.merge(transcripts['Amino Acid Sequence'], left_on = 'Transcript stable ID', right_index = True)
    
    #group information by gene stable ID and amino acid sequence (each unique protein isoform associated with the gene)
    transcripts =  merged.groupby(['Gene stable ID','Gene name', 'Amino Acid Sequence'])['Transcript stable ID'].apply(set).apply(','.join)
    proteins = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['UniProtKB/Swiss-Prot ID'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
    isoform_id = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['UniProtKB isoform ID'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
    canonical = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['Uniprot Canonical'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
    
    #combine into dataframe
    isoforms = pd.concat([proteins, isoform_id, canonical, transcripts], axis = 1).reset_index()
    isoforms['Uniprot Canonical'] = isoforms['Uniprot Canonical'].replace('', 'Ensembl Alternative')
    return isoforms


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
    







