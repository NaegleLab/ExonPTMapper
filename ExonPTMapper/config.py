import pandas as pd
import numpy as np

#file processing packages
import json
import sys
import os
import warnings
import logging
import datetime

from proteomeScoutAPI import ProteomeScoutAPI


package_dir = os.path.dirname(os.path.abspath(__file__))
modification_conversion = pd.read_csv(package_dir + '/../Resource_Files/modification_conversion.csv')


#update these lines as needed
#api_dir = 'C:\\Users\\crowl\\OneDrive\\Documents\\GradSchool\\Research\\ProteomeScoutAPI\\'
#ps_data_dir = api_dir + '/ProteomeScoutAPI/Proteome/data.tsv'
source_data_dir = './source_data/'
processed_data_dir = './processed_data/'
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
#sys.path.append(api_dir)
#from proteomeScoutAPI import ProteomeScoutAPI
ps_api = ProteomeScoutAPI()

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
print('Downloading UniProt isoforms information')
if os.path.isfile(source_data_dir + 'uniprot_canonical_ids.json') and os.path.isfile(source_data_dir + 'uniprot_isoforms_ids.json'):
    with open(source_data_dir + 'uniprot_canonical_ids.json', 'r') as f:
        canonical_isoIDs = json.load(f)
    with open(source_data_dir + 'uniprot_isoforms_ids.json', 'r') as f:
        all_isoforms = json.load(f)
        #iterate through and separate by semicolon
        for key, value in all_isoforms.items():
            if value == value:
                all_isoforms[key] = value.split(';')
else:
    from ExonPTMapper import utility
    canonical_isoIDs, all_isoforms = utility.get_uniprot_isoform_info()

    #join all_isoforms list into string separated by ';'
    for key, value in all_isoforms.items():
        if value == value:
            all_isoforms[key] = ';'.join(value)

    #save dictionary as json file
    with open(source_data_dir + 'uniprot_canonical_ids.json', 'w') as f:
        json.dump(canonical_isoIDs, f)

    with open(source_data_dir + 'uniprot_isoforms_ids.json', 'w') as f:
        json.dump(all_isoforms, f)


#load uniprot translator dataframe, process if need be
print('Downloading ID translator file')
if os.path.isfile(source_data_dir + 'translator.csv'):
    translator = pd.read_csv(source_data_dir + 'translator.csv')
else:
    logger.info('Translator file not found. Downloading from Database IDs of Ensembl, UniProt, PDB, CCDS, and Refseq via pybiomart.')
    from ExonPTMapper import utility
    translator = utility.download_translator(logger, canonical_isoIDs)
    logger.info('Finished downloading and processing translator file. Saving to processed data directory.')
    
    #save to processed data directory
    translator.to_csv(source_data_dir + 'translator.csv')
    
#if os.path.isfile(processed_data_dir + 'isoforms.csv'):
#    isoforms = pd.read_csv(processed_data_dir + 'isoforms.csv')




