import pandas as pd
import numpy as np

#file processing packages
import json
import sys
import os
import warnings
import logging
import datetime

#bio packages
import pybiomart
from Bio import SeqIO
from ExonPTMapper import utility




#update these lines as needed
api_dir = './'
ps_data_dir = api_dir + '/ProteomeScoutAPI/proteomescout_mammalia_20220131/data.tsv'
source_data_dir = '/source_data/'
processed_data_dir = '/processed_data_dir/'
translator_file = 'uniprot_translator.csv'
available_transcripts_file = processed_data_dir + 'available_transcripts.json'

#initialize logger
logger = logging.getLogger('Configuration')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(processed_data_dir + 'ExonPTMapper.log')
log_format = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(log_format)
#     if (logger.hasHandlers()):
#         logger.handlers.clear()
logger.addHandler(handler)

#load ProteomeScoutAPI
sys.path.append(api_dir)
from ProteomeScoutAPI import proteomeScoutAPI
ps_api = proteomeScoutAPI.ProteomeScoutAPI(ps_data_dir)

#check if available transcripts (transcripts with matching info in UniProt and Ensembl) have been identified. If so, load.
if os.path.isfile(available_transcripts_file):
    with open(available_transcripts_file, 'r') as f:
        available_transcripts = json.load(f)
else:
    print('Indicated available transcript file does not exist. Run processing.getMatchedTranscripts(). \n')
    available_transcripts = None

if os.path.isfile(processed_data_dir + 'pscout_matched_transcripts.json'):
    with open(processed_data_dir + 'pscout_matched_transcripts.json', 'r') as f:
        pscout_matched_transcripts = json.load(f)
else:
    pscout_matched_transcripts = None

if os.path.isfile(processed_data_dir + 'psp_matched_transcripts.json'):
    with open(processed_data_dir + 'psp_matched_transcripts.json', 'r') as f:
        psp_matched_transcripts = json.load(f)
else:
    psp_matched_transcripts = None


#Download the UniProt isoform ids associated with the listed canonical isoform
print('Downloading Canonical UniProt isoforms')
if os.path.isfile(source_data_dir + 'uniprot_canonical_ids.json'):
    with open(source_data_dir + 'uniprot_canonical_ids.json', 'r') as f:
        canonical_isoIDs = json.load(f)
else:
    #start up session for interfacting with rest api
    session, re_next_link = utility.establish_session()

    url =  "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+organism_id:9606&format=tsv&fields=accession,cc_alternative_products&size=500"
    canonical_isoIDs = {}
    for batch, total in utility.get_batch(url, session, re_next_link):
        for line in batch.text.splitlines()[1:]:
            primaryAccession, alternative_products = line.split('\t')
            canonical_isoIDs[primaryAccession] = utility.get_canonical_isoID(alternative_products, primaryAccession)

    #save dictionary as json file
    with open(source_data_dir + 'uniprot_canonical_ids.json', 'w') as f:
        json.dump(canonical_isoIDs, f)


#load uniprot translator dataframe, process if need be
print('Downloading ID translator file')
if os.path.isfile(processed_data_dir + 'translator.csv'):
    translator = pd.read_csv(source_data_dir + 'translator.csv')
else:
    logger.info('Translator file not found. Downloading from Database IDs of Ensembl, UniProt, PDB, CCDS, and Refseq via pybiomart.')
    translator = utility.download_translator(logger)

    #indicate whether listed isoforms are canonical, alternative or unannotated in uniprot
    translator["UniProt Isoform Type"] = np.nan
    translator.loc[translator['UniProtKB isoform ID'].isin(canonical_isoIDs.values()), "UniProt Isoform Type"] = 'Canonical' #if annotated as canonical
    translator.loc[(translator['UniProtKB isoform ID'].isna()) & (~translator['UniProtKB/Swiss-Prot ID'].isna())] = 'Canonical'  #if the only isoform
    translator.loc[(~translator['UniProtKB isoform ID'].isna()) & (translator["UniProt Isoform Type"].isna()), "UniProt Isoform Type"] = 'Alternative'   #if has isoform ID but is not identified as canonical

    logger.info('Finished downloading and processing translator file. Saving to processed data directory.')
    
    #save to processed data directory
    translator.to_csv(source_data_dir + 'translator.csv')
    
#if os.path.isfile(processed_data_dir + 'isoforms.csv'):
#    isoforms = pd.read_csv(processed_data_dir + 'isoforms.csv')




