import pandas as pd
import numpy as np
import json
import gzip
import sys
import os
import pybiomart
import warnings
import logging
import datetime
from Bio import SeqIO
from ExonPTMapper import utility
#import swifter

#update these lines as needed
api_dir = './'
ps_data_dir = './ProteomeScoutAPI/proteomescout_mammalia_20220131/data.tsv'
source_data_dir = './ensembl_data/'
processed_data_dir = './processed_data_dir/'
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
    print('Indicated available transcript file does not exist. Run.... \n')
    available_transcripts = None



#load uniprot translator dataframe, process if need be
print('Downloading ID translator file')
if os.path.isfile(processed_data_dir + 'translator.csv'):
    translator = pd.read_csv(processed_data_dir + 'translator.csv')
else:
    logger.info('Translator file not found. Downloading from Database IDs of Ensembl, UniProt, PDB, CCDS, and Refseq via pybiomart.')
    #restrict to primary chromosomes (avoid haplotypes)
    chromosomes = ['X', '20', '1', '6', '3', '7', '12', '11', '4', '17', '2', '16',
       '8', '19', '9', '13', '14', '5', '22', '10', 'Y', '18', '15', '21',
       'MT']
    
    #initialize pybiomart server/dataset
    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',
                   host='http://www.ensembl.org')
    
    #load ID data that relates Ensembl to UniProt
    translator = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id',
                                       'uniprotswissprot', 'uniprot_isoform'],
             filters = {'biotype':'protein_coding','transcript_biotype':'protein_coding',
             'chromosome_name':chromosomes})
    #identify whether transcript is associated with canonical uniprot id
    translator['Uniprot Canonical'] = translator.apply(utility.is_canonical, axis =1)

    #identify potential gene name errors and report if any (same gene name and transcript ID are associated with multiple protein IDs)
    translator['Warnings'] = np.nan
    utility.checkForTranslatorErrors(translator, logger=logger)

    #load additional ID data that relates Ensembl to external databases (PDB, RefSeq, CCSD)
    translator2 = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id', 'pdb','refseq_mrna', 'ccds'],
             filters = {'biotype':'protein_coding','transcript_biotype':'protein_coding',
             'chromosome_name':chromosomes})
    
    #merge two dataframes, drop any duplicates caused by merging
    translator = translator.merge(translator2, on = ['Gene stable ID', 'Gene name', 'Transcript stable ID'], how = 'left')
    translator = translator.drop_duplicates()

    logger.info('Finished downloading and processing translator file. Saving to processed data directory.')
    
    #save to processed data directory
    translator.to_csv(processed_data_dir + 'translator.csv')
    
#if os.path.isfile(processed_data_dir + 'isoforms.csv'):
#    isoforms = pd.read_csv(processed_data_dir + 'isoforms.csv')










