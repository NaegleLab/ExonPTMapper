import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import json
import gzip
import re
import sys
import logging
import math
from collections import defaultdict
import pybiomart
from ExonPTMapper import config
import time

#import swifter

#initialize logger
logger = logging.getLogger('Processing')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(config.processed_data_dir + 'ExonPTMapper.log')
log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(log_format)
logger.addHandler(handler)

def downloadMetaInformation(gene_attributes = ['ensembl_gene_id','external_gene_name', 'strand','start_position','end_position', 'chromosome_name', 'uniprotswissprot'],
                            transcript_attributes = ['ensembl_gene_id','ensembl_transcript_id','transcript_length','transcript_appris', 'transcript_is_canonical','transcript_tsl'],
                            exon_attributes = ['ensembl_gene_id', 'ensembl_transcript_id', 'ensembl_exon_id', 'is_constitutive','rank','exon_chrom_start', 'exon_chrom_end'],
                            filters = {'transcript_biotype':'protein_coding', 'transcript_gencode_basic':True, 
                                       'chromosome_name': ['X', 'Y', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22','MT']}):
    """
    Using pybiomart, download basic information for all protein-coding genes, transcripts, and exons. Will both save the dataframes to the processed_data_dir and return them as a tuple.

    Parameters:
    ----------
    gene_attributes: list
        Attributes to download for each gene. Default is ['ensembl_gene_id','external_gene_name', 'strand','start_position','end_position', 'chromosome_name', 'uniprotswissprot']. For list of possible attributes, use the dataset.list_attributes() method within pybiomart, see their documentation for more details.
    transcript_attributes: list
        Attributes to download for each transcript. Default is ['ensembl_gene_id','ensembl_transcript_id','transcript_length','transcript_appris', 'transcript_is_canonical','transcript_tsl']. For list of possible attributes, use the dataset.list_attributes() method within pybiomart, see their documentation for more details.
    exon_attributes: list
        Attributes to download for each exon. Default is ['ensembl_gene_id', 'ensembl_transcript_id', 'ensembl_exon_id', 'is_constitutive','rank','exon_chrom_start', 'exon_chrom_end']. For list of possible attributes, use the dataset.list_attributes() method within pybiomart, see their documentation for more details.
    filters: dict
        Filters that restrict the information being downloaded, namely to be protein coding and a part of the gencode basic set. Default is {'biotype':'protein_coding','transcript_biotype':'protein_coding', 'transcript_gencode_basic':True}. All exon information will be downloaded, and then filtered based on transcript/gene info. For list of possible filters, use the dataset.list_filters() method within pybiomart, see their documentation for more details.

    Returns:
    ----------
    genes: pandas.DataFrame
        DataFrame containing gene-specific information. Index is Ensembl Gene ID, columns are attributes specified in gene_attributes.
    transcripts: pandas.DataFrame
        DataFrame containing transcript-specific information. Index is Ensembl Transcript ID, columns are attributes specified in transcript_attributes.
    exons: pandas.DataFrame
        DataFrame containing exon-specific information. Columns are attributes specified in exon_attributes. Because a single exon may be found in multiple transcripts, each row does not necesarily correspond to a unique exon, but does contain a unique exon/transcript pair.
    
    """  
    logger.info('Downloading exon, transcript and gene meta information from Ensembl via pybiomart')    
    #initialize ensembl dataset, restrict to primary chromosome information (no haplotypes)
    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',host='http://www.ensembl.org')
    
    #download gene info
    print('Downloading and processing gene-specific meta information')
    genes = dataset.query(attributes=gene_attributes,filters = filters)
    #identify all genes with uniprot id and collapse into one row for each gene
    genes = collapseGenesByProtein(genes)

    #save data
    print('saving\n')
    genes.to_csv(config.processed_data_dir + 'genes.csv')
    
    #download transcript info
    print('Downloading and processing transcript-specific meta information')
    transcripts = dataset.query(attributes=transcript_attributes,
                 filters = filters)
    transcripts.index = transcripts['Transcript stable ID']
    transcripts = transcripts.drop('Transcript stable ID', axis = 1)
    #transcripts = transcripts.drop_duplicates()
    #transcripts = transcripts[transcripts['Gene stable ID'].isin(genes.index)]
    print('saving\n')
    transcripts.to_csv(config.processed_data_dir + 'transcripts.csv')

    #download exon info
    print('Downloading and processing exon-specific meta information')
    exons = dataset.query(attributes=exon_attributes,
                 filters = filters)
    exons = exons.rename({'Exon region start (bp)':'Exon Start (Gene)', 'Exon region end (bp)':'Exon End (Gene)'}, axis = 1)
    print('saving\n')
    exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)

    #check to make sure same information is found in each dataframe

    if exons['Transcript stable ID'].nunique() == transcripts.shape[0]:
        logger.warning('Different number of transcripts found in exon and transcript dataframe. This may be due to transcripts being filtered out due to missing information. Proceed with caution.')

    logger.info('Exon, transcript, and gene information saved in processed data directory. A total of {} genes, {} transcripts, and {} exons were downloaded.'.format(genes.shape[0], transcripts.shape[0], exons['Exon stable ID'].nunique()))
    
    return genes, transcripts, exons



def processExons(exon_info, exon_sequences):
    """
    Given an exon information dataframe and a dataframe containing the associated exon sequences, perform the following tasks:
    1. Combine exon meta information and sequence information into one dataframe
    2. Remove transcripts with incomplete exon sequence information. This will be the case if not all exons are found in the exon dataframe, or if the exon sequence is not found in the exon sequence dataframe.
    3. Using exon rank information and total transcript length, determine the start and end position of each exon in the transcript sequence.

    Parameters:
    ----------
    exon_info: pandas.DataFrame
        Dataframe containing exon meta information, which is also the same dataframe downloaded by downloadMetaInformation()
    exon_sequences: pandas.DataFrame
        Dataframe containing exon sequences obtained from BioMart. Must contain the columns 'Exon stable ID' and 'Exon Sequence'

    Returns:
    ----------
    exons: pandas.DataFrame
        Updated exon dataframe which contains exon sequences, lengths, and location in transcript sequence.
    """
    logger.info('Adding sequence information to exon dataframe')

    #check to see what exons did not have sequence information in sequence data
    missing_seq_exons = set(exon_info['Exon stable ID']).difference(set(exon_sequences['Exon stable ID']))
    if len(missing_seq_exons) > 0:
        logger.warning(f'{len(missing_seq_exons)} exons did not have sequence information in sequence data. Transcripts containing these exons will be removed from analysis. The following exons do not have sequence information: {", ".join(missing_seq_exons)}.')


    #load exon specific information and compile into one data structure
    exons = pd.merge(exon_info, exon_sequences, on ='Exon stable ID', how = 'left')

    #check to make sure exon dataframe is the same size as before
    if exon_info.shape[0] != exons.shape[0]:
        logger.warning('Merging exon information and exon sequence information resulted in duplicate rows. Removing duplicates, but proceed with caution')
        exons = exons.drop_duplicates()
    
    #remove transcripts with incomplete exon sequence information
    missing_info_transcripts = exons.loc[exons['Exon Sequence'].isna()]
    #if it is the first exon or last exon in sequence, that is okay
    remove_transcripts = []
    for i, row in missing_info_transcripts.iterrows():
        if row['Exon rank in transcript'] != 1 and row['Exon rank in transcript'] != exons.loc[exons['Transcript stable ID'] == row['Transcript stable ID'], 'Exon rank in transcript'].max():
            remove_transcripts.append(row['Transcript stable ID'])
    remove_transcripts = np.unique(remove_transcripts)
    if len(remove_transcripts) > 0:
        print(f'{len(remove_transcripts)} with incomplete exon sequence information. Removing from analysis.')
        logger.warning(f'{len(remove_transcripts)} transcripts with incomplete exon sequence information. Removing from analysis.')
    exons = exons[~exons['Transcript stable ID'].isin(remove_transcripts)]
    exons = exons.dropna(subset = 'Exon Sequence')

    print('Getting exon lengths and cuts')
    logger.info('Getting exon lengths and locations in transcript sequence')
    start = time.time()
    #get exon lengths from sequence information
    exons["Exon Length"] = exons.apply(exonlength, axis = 1)
    #sort exons by exon rank, and then identify the start and end position of each exon in the transcript sequence by summing the exon lengths
    exons = exons.sort_values(by = ['Transcript stable ID', 'Exon rank in transcript'])
    exons['Exon End (Transcript)'] = exons.groupby('Transcript stable ID').cumsum(numeric_only = True)['Exon Length']
    exons['Exon Start (Transcript)'] = exons['Exon End (Transcript)'] - exons['Exon Length']
    end = time.time()
    print('Elapsed time:', end - start, '\n')
    
    return exons
    
def processTranscripts(transcripts, coding_seqs, exons, APPRIS = None):
    """
    Given the transcript dataframe from downloadMetaInformation(), add the following information:
    1. Transcript sequence, based on exon sequence information
    2. Coding sequence of the transcript, obtained from the coding sequence dataframe. Also note the start and end position of the coding sequence in the transcript sequence.
    3. IF provided, add TRIFID functional scores obtained from the APPRIS database
    4. Annotate the transcript with whether it is associated with UniProt canonical or alternative ID

    Parameters
    ----------
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from downloadMetaInformation()
    coding_seqs: pandas.DataFrame
        Coding sequences downloaded from BioMart and converted into series object
    exons: pandas.DataFrame
        Exon dataframe obtained from processExons()
    APPRIS: pandas.DataFrame
        Dataframe containing TRIFID functional scores associated with Ensembl transcripts, downloaded from the APPRIS database. Must include the following columns: 'transcript_id','ccdsid','norm_trifid_score'. If not provided, this information will not be added to the transcript dataframe.

    Returns
    ----------
    transcripts: pandas.DataFrame
        Updated transcript dataframe which contains transcript sequences, coding sequences, and TRIFID functional scores (if provided)
    """

    print('Getting transcript sequences from exon data')
    logger.info('Getting complete transcript sequences from exon data')
    start = time.time()
    #sort exons in correct order (by rank in transcript)
    sorted_exons = exons.sort_values(by = ['Transcript stable ID','Exon rank in transcript']).copy()
    #get transcript sequences from exon sequences
    transcript_sequences = sorted_exons.groupby('Transcript stable ID')['Exon Sequence'].apply(''.join)
    transcript_sequences.name = 'Transcript Sequence'
    #get exon cuts/splice boundaries from exon information
    sorted_exons['Exon End (Transcript)'] = sorted_exons['Exon End (Transcript)'].astype(str)
    transcript_cuts = sorted_exons.groupby('Transcript stable ID')['Exon End (Transcript)'].apply(','.join)
    transcript_cuts.name = 'Exon cuts'
    transcripts = transcripts.join([transcript_cuts, transcript_sequences])
    end = time.time()
    print('Elapsed time:',end - start,'\n')

    #store starting shape of transcript dataframe to check for potential errors later
    initial_transcript_shape = transcripts.shape[0]

    print('Finding coding sequence location in transcript')
    logger.info('Finding location of coding sequence in transcript')
    start = time.time()
    #add coding sequences to transcript info
    transcripts = transcripts.join([coding_seqs])
    #find start and end position of coding sequence in transcript sequence, and report any errors
    cds_start, cds_stop, warnings = findCodingRegion(transcripts)
    transcripts['Relative CDS Start (bp)'] = cds_start
    transcripts['Relative CDS Stop (bp)'] = cds_stop
    transcripts['Warnings'] = warnings
    end = time.time()
    print('Elapsed time:',end - start,'\n')

    #record how many transcripts have missing coding sequence information
    missing_cds_loc = transcripts['Warnings'].count()
    print(f'There were {missing_cds_loc} transcripts whose coding sequence location could not be identified ({round(missing_cds_loc/transcripts.shape[0]*100, 2)}%)')
    logger.info(f'There were {missing_cds_loc} transcripts whose coding sequence location could not be identified ({round(missing_cds_loc/transcripts.shape[0]*100, 2)}%)')
    
    #add appris functional scores if information is provided
    if APPRIS is not None:
        print('Adding APPRIS functional scores')
        logger.info('Adding data downloaded from APPRIS database, including TRIFID functional scores')
        start = time.time()
        #extract relevant columns and rename to more readable column names
        APPRIS = APPRIS[['transcript_id','ccdsid','norm_trifid_score']]
        APPRIS = APPRIS.rename({'transcript_id':'Transcript stable ID', 'ccdsid':'CCDS ID', 'norm_trifid_score':'TRIFID Score'}, axis = 1)
        #check to make sure there are not duplicate transcript entries
        if APPRIS['Transcript stable ID'].nunique() != APPRIS.shape[0]:
            logger.warning('APPRIS data contains rows with duplicate transcripts. Removing duplicates, but proceed with caution')
            APPRIS = APPRIS.drop_duplicates(subset = 'Transcript stable ID')
        #add appris data to transcripts dataframe
        transcripts = transcripts.merge(APPRIS, on = 'Transcript stable ID', how = 'left')
        end = time.time()
        print('Elapsed time:', end-start,'\n')
    
    print('Getting amino acid sequences')
    #translate coding region to get amino acid sequence using biopython
    start = time.time()
    transcripts = transcripts.apply(translate, axis = 1)
    end = time.time()
    
    #return the fraction of transcripts that were successfully translated: some may not due to missing coding sequences, coding sequences that are not multiples of 3, or other issues that arose during translation
    fraction_translated = transcripts.dropna(subset = 'Amino Acid Sequence').shape[0]/transcripts.shape[0]
    logger.info(f'Fraction of transcripts that were successfully translated: {round(fraction_translated, 3)}')
    print(f'Fraction of transcripts that were successfully translated: {round(fraction_translated, 3)}')
    print('Elapsed time:',end -start,'\n')

    
    #indicate whether transcript is canonical based on translator dataframe
    trim_translator = config.translator[['Transcript stable ID', 'Uniprot Canonical']].drop_duplicates()
    transcripts = transcripts.merge(trim_translator, on = 'Transcript stable ID', how = 'left')
    
    #add transcript id as index again
    transcripts.index = transcripts['Transcript stable ID']
    transcripts = transcripts.drop('Transcript stable ID', axis = 1)

    #make sure transcript 
    if initial_transcript_shape != transcripts.shape[0]:
        logger.warning(f'Size of transcripts dataframe changed during processing. Initial size was {initial_transcript_shape}, final size is {transcripts.shape[0]}. Removing any duplicate rows, but proceed with caution')
    
    return transcripts

def getIsoformInfo(transcripts):
    """
    Get dataframe where each row corresponds to a unique protein isoform, with transcripts with identitical protein information are collapsed into the same row with a unique isoform identifier. This identifier is either the associated UniProtKB isoform ID, or, if not associated with a UniProtKB protein, a unique identifier with the format 'ENS-{Gene Name}-{Isoform Number}'. 

    Parameters
    ----------
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from processTranscripts()

    Returns
    ----------
    isoforms: pandas.DataFrame
        Isoform-specific dataframe, in which each row corresponds to a unique protein isoform sequence. Includes information about which transcripts are associated with each isoform identifier and the amino acid sequence
    """
    logger.info('Creating isoform dataframe by collapsing transcripts with identical protein sequences')
    #add amino acid sequence to translator information
    isoforms = transcripts[['Gene stable ID', 'Amino Acid Sequence']]
    #group transcripts by gene and protein sequences
    isoforms = isoforms.reset_index().groupby(['Amino Acid Sequence', 'Gene stable ID'])['Transcript stable ID'].agg(';'.join).reset_index()
    
    #iterate through each grouped isoform, if it has uniprot id, use that, else make up new name that follows this format 'ENS_{Gene Name}_{Isoform_num}'
    isoform_id = []
    isoform_type = []
    isoform_numbers = defaultdict(int)
    for i, row in isoforms.iterrows():
        #grab tranlator information for transcripts associated with isoform
        tmp = config.translator[config.translator['Transcript stable ID'].isin(row['Transcript stable ID'].split(';'))]
        #check if isoform has uniprot id associated with any of the transcripts, if not, create new isoform id
        if tmp['UniProtKB isoform ID'].isna().all() and tmp['UniProtKB/Swiss-Prot ID'].isna().all():
            gene_name = tmp['Gene name'].unique()[0]
            isoform_numbers[gene_name] = isoform_numbers[gene_name] + 1
            isoform_id.append(f'ENS-{gene_name}-{isoform_numbers[gene_name]}')
            isoform_type.append('Alternative')
        elif not tmp['UniProtKB isoform ID'].isna().all():
            tmp = tmp.dropna(subset = 'UniProtKB isoform ID')
            isoform_id.append(tmp['UniProtKB isoform ID'].unique()[0])
            if '-1' in tmp['UniProtKB isoform ID'].unique()[0]:
                isoform_type.append('Canonical')
            else:
                isoform_type.append('Alternative')
        else:
            tmp = tmp.dropna(subset = 'UniProtKB/Swiss-Prot ID')
            isoform_id.append(tmp['UniProtKB/Swiss-Prot ID'].unique()[0] + '-1')
            isoform_type.append('Canonical')
    isoforms['Isoform ID'] = isoform_id
    isoforms['Isoform Type'] = isoform_type
    #get length of each isoform
    isoforms['Isoform Length'] = isoforms['Amino Acid Sequence'].apply(len)
    
    logger.info('Isoform dataframe created.')
    return isoforms

def getProteinInfo(transcripts, genes):
    """
    Process translator dataframe so to get protein specific information (collapse protein isoforms into one row). Add context, including:
    1. Transcripts associated with the canonical uniprot protein
    2. Number of alternative isoforms in UniProt
    3. Alternative transcripts associated with UniProt isoforms
    4. Transcripts associated with the canonical uniprot protein AND with matching protein sequence (same for alternative transcripts)
    5. All transcripts associated with the gene, regardless of if it relates to a UniProt entry
    5. Whether the gene associated with the protein is unique (i.e. only one protein associated with this gene) 

    Parameters
    ----------
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from processTranscripts()
    genes: pandas.DataFrame
        Gene dataframe obtained from downloadMetaInformation()

    Returns
    ----------
    proteins: pandas.DataFrame
        Protein-specific dataframe, in which each row corresponds to a unique UniProt protein. Includes information about the number of uniprot isoforms and transcripts associated with canonical or alternative isoforms
    """
    logger.info('Constructing protein-specific dataframe')
    #grab canonical UniProt proteins and associated transcripts from translator dataframe
    proteins = config.translator[config.translator['Uniprot Canonical'] == 'Canonical']
    proteins = proteins[['UniProtKB/Swiss-Prot ID','Transcript stable ID']].drop_duplicates().copy()
    #aggregate information along unique UniProt IDs
    proteins = proteins.groupby('UniProtKB/Swiss-Prot ID').agg(';'.join)
    proteins.columns = ['Canonical Transcripts']

    #repeat above process but for alternative UniProt isoforms: get transcripts and number of isoforms
    variants = config.translator[config.translator['Uniprot Canonical'] == 'Alternative']
    variants = variants[['UniProtKB/Swiss-Prot ID', 'Transcript stable ID']].drop_duplicates()
    variants_grouped = variants.groupby('UniProtKB/Swiss-Prot ID')
    num_variants = variants_grouped.count() + 1
    num_variants.columns = ['Number of Uniprot Isoforms']
    variant_trans = variants_grouped.agg(';'.join)
    variant_trans.columns = ['Alternative Transcripts (Uniprot Isoforms)']

    #add available canonical transcripts with matching uniprot sequence (check if transcript is found in config.available_transcripts list)
    if config.available_transcripts is not None:
        canonical_matches = []
        for trans in proteins['Canonical Transcripts']:
            match = []
            for available in config.available_transcripts:
                if available in trans:
                    match.append(available)
            #add to list of matches (if no matches, add np.nan)
            if len(match) > 0:
                canonical_matches.append(';'.join(match))
            else:
                canonical_matches.append(np.nan)
        proteins['Matched Canonical Transcripts'] = canonical_matches
    else:
        logger.warning('No list of available transcripts provided. Cannot determine which transcripts have matching sequences to UniProt canonical isoforms. Run processing.getMatchedTranscripts() to get this information.')
        print('No list of available transcripts provided. Cannot determine which transcripts have matching sequences to UniProt canonical isoforms. Run processing.getMatchedTranscripts() to get this information.')

    #add number of uniprot isoforms to dataframe, replace nan with 1 (only have the canonical)
    proteins = proteins.merge(num_variants, left_index = True, right_index = True, how = 'outer')
    proteins.loc[proteins['Number of Uniprot Isoforms'].isna(), 'Number of Uniprot Isoforms'] = 1

    #add alternative transcripts to dataframe
    proteins = proteins.merge(variant_trans, left_index = True, right_index = True, how = 'outer')

    #add available transcripts with matching uniprot sequence
    if config.available_transcripts is not None:
        alternative_matches = []
        for trans in proteins['Alternative Transcripts (Uniprot Isoforms)']:
            match = []
            if trans is np.nan:
                alternative_matches.append(np.nan)
            else:
                for available in config.available_transcripts:
                    if available in trans:
                        match.append(available)
                #add to list of matches (if no matches, add np.nan)
                if len(match) > 0:
                    alternative_matches.append(';'.join(match))
                else:
                    alternative_matches.append(np.nan)
        proteins['Matched Alternative Transcripts'] = alternative_matches
    
    #grab Ensembl gene ids associated with each uniprot protein, explode on each gene id
    prot_genes = config.translator.groupby('UniProtKB/Swiss-Prot ID')['Gene stable ID'].apply(set)
    proteins['Gene stable IDs'] = prot_genes.apply(';'.join)
    prot_genes = prot_genes.explode().reset_index()

    #grab all transcripts associated with the gene, that are not associated with the canonical uniprot protein
    alt_transcripts = config.translator[config.translator['Uniprot Canonical'] != 'Canonical'].groupby('Gene stable ID')['Transcript stable ID'].apply(';'.join).reset_index()
    prot_genes = pd.merge(prot_genes,alt_transcripts, on = 'Gene stable ID', how = 'left')

    #grab genes for which there are multiple uniprot proteins associated with it, add to prot_genes info
    nonunique_genes = genes[genes['Number of Associated Uniprot Proteins'] > 1].index
    nonunique_genes = pd.DataFrame({'Unique Gene':np.repeat('No', len(nonunique_genes)), 'Gene stable ID':nonunique_genes})
    prot_genes = prot_genes.merge(nonunique_genes, on = 'Gene stable ID', how = 'left')
    prot_genes.index = prot_genes['UniProtKB/Swiss-Prot ID']
    prot_genes = prot_genes.drop('UniProtKB/Swiss-Prot ID', axis = 1)
    #For proteins associated with unique gene, mark as unique
    prot_genes['Unique Gene'] = prot_genes['Unique Gene'].replace(np.nan, 'Yes')


    #add all transcripts associated with the gene, regardless of if it is associated with a uniprot protein
    proteins['Alternative Transcripts (All)'] = prot_genes.dropna(subset = 'Transcript stable ID').groupby('UniProtKB/Swiss-Prot ID')['Transcript stable ID'].apply(set).apply(';'.join)
    proteins['Unique Gene'] = prot_genes.groupby('UniProtKB/Swiss-Prot ID')['Unique Gene'].apply(set).apply(';'.join)

    #report how many proteins are not associated with a unique gene
    num_nonunique = prot_genes[prot_genes['Unique Gene'] == 'No'].shape[0]
    print(f'{num_nonunique} proteins are associated with a gene that is not unique (i.e. multiple proteins are associated with the same gene).')
    logger.warning(f'{num_nonunique} proteins are associated with a gene that is not unique (i.e. multiple proteins are associated with the same gene).')
    
    return proteins
    
    
def collapseGenesByProtein(genes):
    """
    Given the gene dataframe downloaded via pybiomart, collapse genes into a single row for each gene, and add the following information:
    1. Number of distinct uniprot entries associated with the gene
    2. All UniProt IDs associated with the gene, separated by a comma
    3. Remove 'UniProtKB/Swiss-Prot' column downloaded from biomart

    Intended for use in the downloadMetaInformation() function

    Parameters
    ----------
    genes: pandas.DataFrame
        Gene dataframe downloaded from pybiomart 
    
    Returns
    ----------
    genes: pandas.DataFrame
        Updated genes dataframe with each row being specific to a single Ensembl Gene ID

    """
    #calculate the number of distinct uniprot entries per gene (exclude nan entries then count unique entries)
    num_uniprot = config.translator.dropna(subset = 'UniProtKB/Swiss-Prot ID').groupby('Gene stable ID')['UniProtKB/Swiss-Prot ID'].nunique()
    num_uniprot.name = 'Number of Associated Uniprot Proteins'
    
    #get the isoform ids
    proteins_from_gene = genes.dropna(subset = 'UniProtKB/Swiss-Prot ID').groupby('Gene stable ID')['UniProtKB/Swiss-Prot ID'].apply(';'.join)
    proteins_from_gene.name = 'Associated Uniprot Proteins'
    
    genes.index = genes['Gene stable ID']
    genes = genes.drop('Gene stable ID', axis = 1)
    genes = genes.join([num_uniprot, proteins_from_gene])
    #remove single protein id column from dataframe and drop duplicates
    genes = genes.drop('UniProtKB/Swiss-Prot ID', axis = 1)
    genes = genes.drop_duplicates()
    return genes

def findCodingRegion(transcripts, transcript_sequence_col = 'Transcript Sequence', coding_sequence_col = 'Coding Sequence'):
    """
    Given the transcripts dataframe containing both the full coding sequence and the transcript sequence, find the start and end position of the coding sequence in the transcript sequence. If the coding sequence is not found in the transcript sequence, return np.nan for both start and end position, and specify the reason in the warnings list

    Parameters
    ----------
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from downloadMetaInformation() with both transcript and coding sequence information added
    transcript_sequence_col: str
        Name of column in transcripts dataframe that contains the transcript sequence. Default is 'Transcript Sequence'
    coding_sequence_col: str
        Name of column in transcripts dataframe that contains the coding sequence. Default is 'Coding Sequence'
    
    Returns
    ----------
    cds_start: list
        List of start positions of the coding sequence in the transcript sequence, in the same order as the inputted transcripts dataframe. Based on pythonic coordinates (i.e 0 is the first base pair in the transcript). If the coding sequence is not found in the transcript sequence, return np.nan.
    cds_end: list
        List of end positions of the coding sequence in the transcript sequence, in the same order as the inputted transcripts dataframe. Based on pythonic coordinates (i.e 0 is the first base pair in the transcript and end coordinate is exclusive). If the coding sequence is not found in the transcript sequence, return np.nan.
    warnings: list
        Any errors/warnings that arose during the process of finding coding sequence location in transcript sequence. Value will be np.nan for transcripts where the coding sequence was found in the transcript sequence.
    """
    #initialize lists to add as new columns to transcript dataframe
    cds_start = []
    cds_end = []
    warnings = []
    #iterate through each transcript, find the start and end position of coding sequence using re.search()
    for i in transcripts.index:
        #grab transcript and coding sequence
        coding_sequence = transcripts.at[i,coding_sequence_col]
        full_transcript = transcripts.at[i,transcript_sequence_col]
        #check to make sure both coding sequence and transcript sequence are available
        if full_transcript is not np.nan and coding_sequence is not np.nan:
            #find start and end position of coding sequence in transcript sequence, if not found, return np.nan
            match = re.search(coding_sequence, full_transcript)
            if match:
                cds_start.append(match.span()[0])
                cds_end.append(match.span()[1])
                warnings.append(np.nan)
            else:
                cds_start.append(np.nan)
                cds_end.append(np.nan)
                warnings.append('Coding Sequence Not Found in Transcript')
        elif coding_sequence is not np.nan:
            cds_start.append(np.nan)
            cds_end.append(np.nan)
            warnings.append('Missing Transcript Sequence Information')
        else:
            cds_start.append(np.nan)
            cds_end.append(np.nan)
            warnings.append('Missing Coding Sequence Information')
    return cds_start, cds_end, warnings

def getMatchedTranscripts(transcripts, update = False): 
    """
    Given the transcript dataframe from processTranscripts() function, identify which canonical transcripts (defined by UniProt) have a matching sequence in the proteomeScout database. This is done by comparing the amino acid sequence of the transcript to the amino acid sequence of the protein in proteomeScout. Transcripts with matching information will be stored in config.available_transcripts and saved to a .json file for future use. Only these transcripts will be considered during mapping.

    Parameters
    ----------
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from processTranscripts()
    update: bool
        Whether to rerun analysis even if there is already a list of available transcripts. Default is False, which will only run analysis if there is no list of available transcripts.

    Returns
    ----------
    None, but will save the list of available transcripts to config.available_transcripts and a .json file in the processed_data_dir
    """
    if config.available_transcripts is None or update:  #check if need to run analysis
        print('Finding available transcripts')
        logger.info('Finding transcripts with matching protein sequence information in ProteomeScout')
        start = time.time()
        #get all transcripts associated with a canonical UniProt ID
        seq_align = config.translator.loc[config.translator['Uniprot Canonical'] == 'Canonical', ['Transcript stable ID', 'UniProtKB/Swiss-Prot ID']].drop_duplicates().copy()
        #record the total number of canonical transcripts
        num_transcripts = seq_align['Transcript stable ID'].nunique()
        num_proteins = seq_align['UniProtKB/Swiss-Prot ID'].nunique()
        
        #determine if ensembl and proteomescout information matches
        seq_align['PS Seq'] = seq_align.apply(get_ps_seq, axis= 1)
        seq_align['GENCODE Seq'] = get_gencode_seq(seq_align, transcripts)
        seq_align['Exact Match'] = seq_align.apply(perfect_align, axis=1)

        #extract the transcripts that have matching sequence information
        perfect_matches = seq_align[seq_align['Exact Match']==True]
        
        #indicate how many transcripts matched
        print(f'{num_transcripts} found associated with canonical UniProt proteins.\n {round(perfect_matches.shape[0]/num_transcripts*100, 2)}% of these transcripts match sequence information in ProteomeScout.')
        logger.info(f'{num_transcripts} found associated with canonical UniProt proteins ({num_proteins} unique proteins).\n {round(perfect_matches.shape[0]/num_transcripts*100, 2)}% of these transcripts match sequence information in ProteomeScout.')
        
        #save the list of available transcripts with matching sequence information
        config.available_transcripts = perfect_matches['Transcript stable ID'].tolist()
        with open(config.processed_data_dir+"available_transcripts.json", 'w') as f:
            json.dump(config.available_transcripts, f, indent=2) 
        end = time.time()
        print('Elapsed time:',end-start, '\n')
    else:
        print('Already have the available transcripts. If you would like to update analysis, set update=True')

def getExonCodingInfo(exon, transcripts, strand):
    """
    Given the processed exon and transcript information, extract the location of the exon in the protein sequence. This includes checking for ragged sites: residues that are only partially encoded by the exon.

    Parameters
    ----------
    exon: pandas.Series
        row of the processed exon dataframe, or series object containing exon information
    transcripts: pandas.DataFrame
        processed transcript dataframe produced by processTranscripts()
    strand: int
        DNA strand that exon is located in: 1 indicates forward strand, -1 indicates reverse strand

    Returns
    -------
    aa_seq_ragged: str
        Amino acid sequence encoded for by exon of interest, including ragged site residues. Ragged site residues are separated with '-'.
    aa_seq_nr: str
        Amino acid sequence encoded for by exon of interest, excluding ragged site residues.
    exon_prot_start: float
        Location of the start of the exon in the protein sequence, using amino acid coordinates (first residue is at position 0). This means that fractional values indicate cases in which the exon partially encodes for a residue
    exon_prot_end: str
        Location of the end of the exon in the protein sequence, using amino acid coordinates (first residue is at position 0). This means that fracitonal values indicate cases in which the exon partially encodes for a residue.
    exon_coding_start: int
        Location of the start of coding sequence in the exon, based on genomic coordinates. This helps indicate whether exon is fully, partially or noncoding (np.nan if noncoding)
    exon_coding_end: int
        Location of the end of the coding sequence in the exon, based on genomic coordinates. This helps indicate whether the exon is fully, partially, or fully noncoding (np.nan if noncoding)
    warnings: str
        If unable to get amino acid sequence, reasons will be reported here. These could be: exon is in a noncoding region, missing transcript info, missing coding sequence, missing transcript sequence, failure to translate transcript

    """
    #make sure exon has associated transcript, if not return Missing Transcript Info
    try:
        transcript = transcripts.loc[exon['Transcript stable ID']]
    except KeyError:
        return tuple(np.repeat(np.nan, 6)) + ('Missing Transcript Info',)
    
    #look for warnings in transcript dataframe
    warnings = transcript['Warnings']
    if warnings == warnings:
        return tuple(np.repeat(np.nan, 6)) + (warnings,)
    else:
        full_aa_seq = transcript['Amino Acid Sequence']
        #check to where exon starts in coding sequence
        if int(exon['Exon End (Transcript)']) <= int(transcript['Relative CDS Start (bp)']):
            return tuple(np.repeat(np.nan, 6)) + ("5' NCR",)
        elif int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']) == 1 or int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']) == 2:      #for rare case where exon only partially encodes for the starting amino acid
            return full_aa_seq[0]+'*', np.nan, np.nan, np.nan, np.nan, np.nan, 'Partial start codon only'
        elif exon['Exon Start (Transcript)'] <= int(transcript['Relative CDS Start (bp)']): #if exon starts before coding sequence, then exon starts at protein start
            exon_prot_start = 0.0
            if strand == 1:     #forward strand, count from start of exon in the gene
                exon_coding_start = (int(transcript['Relative CDS Start (bp)']) - exon['Exon Start (Transcript)']) + exon['Exon Start (Gene)']
            else:   #reverse strand, count back from end of exon in the gene
                exon_coding_start = exon['Exon End (Gene)'] - (int(transcript['Relative CDS Start (bp)']) - exon['Exon Start (Transcript)'])
        else:
            exon_prot_start = (exon['Exon Start (Transcript)'] - int(transcript['Relative CDS Start (bp)']))/3
            exon_coding_start = exon['Exon Start (Gene)'] if strand == 1 else exon['Exon End (Gene)']
        
        #check to see where exon ends in the end of coding sequence
        exon_prot_end = (exon['Exon End (Transcript)'] - int(transcript['Relative CDS Start (bp)']))/3
        if exon['Exon Start (Transcript)'] > int(transcript['Relative CDS Start (bp)'])+len(transcript['Coding Sequence']):  #if exon starts after the end of the coding sequence
            return tuple(np.repeat(np.nan, 6)) + ("3' NCR",)
        # in some cases a stop codon is present in the middle of the coding sequence: this is designed to catch those cases (also might be good to identify these cases)
        elif exon['Exon Start (Transcript)'] > int(transcript['Relative CDS Start (bp)'])+len(transcript['Amino Acid Sequence'])*3:   #if exon starts after the end of the coding sequence (based on amino acid sequence)
            return  tuple(np.repeat(np.nan, 6)) + ("3' NCR",)
        elif exon_prot_end > float(len(transcript['Amino Acid Sequence'])): #if exon contains the end of the coding sequence
            exon_prot_end= float(len(transcript['Amino Acid Sequence']))
            if strand == 1:
                exon_coding_end = exon['Exon Start (Gene)'] + (int(transcript['Relative CDS Stop (bp)']) - exon['Exon Start (Transcript)'])
            else:
                exon_coding_end = exon['Exon Start (Gene)'] + (exon['Exon End (Transcript)'] - int(transcript['Relative CDS Stop (bp)']))
        else:
            exon_prot_end= (int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']))/3 
            exon_coding_end = exon['Exon End (Gene)'] if strand == 1 else exon['Exon Start (Gene)']

        warnings = np.nan
        #based on locations of start and end of exon protein sequence, identify cases where the exon encodes for a ragged amino acid (at splice boundary, so exon only partially encodes for amino acid, the rest is coded for by a different exon). If protein start and end are not integers, then the exon encodes for a ragged amino acid. Here, get exon sequences that both include and exclude these ragged sequences
        if exon_prot_start.is_integer() and exon_prot_end.is_integer(): #no ragged residues
            aa_seq_ragged = full_aa_seq[int(exon_prot_start):int(exon_prot_end)]
            aa_seq_nr = full_aa_seq[int(exon_prot_start):int(exon_prot_end)]
        elif exon_prot_end.is_integer(): #ragged residues at start of exon
            ragged_start = math.floor(exon_prot_start)
            full_start = math.ceil(exon_prot_start)
            aa_seq_ragged = full_aa_seq[ragged_start]+'-'+full_aa_seq[full_start:int(exon_prot_end)]
            aa_seq_nr = full_aa_seq[full_start:int(exon_prot_end)]
        elif exon_prot_start.is_integer(): #ragged residues at end of exon
            ragged_stop = math.ceil(exon_prot_end)
            full_stop = math.floor(exon_prot_end)
            aa_seq_ragged = full_aa_seq[int(exon_prot_start):full_stop]+'-'+full_aa_seq[ragged_stop-1]
            aa_seq_nr = full_aa_seq[int(exon_prot_start):full_stop]
        else: #ragged residues at both start and end of exon
            ragged_start = math.floor(exon_prot_start)
            full_start = math.ceil(exon_prot_start)
            ragged_stop = math.ceil(exon_prot_end)
            full_stop = math.floor(exon_prot_end)
            aa_seq_ragged = full_aa_seq[ragged_start]+'-'+full_aa_seq[full_start:full_stop]+'-'+full_aa_seq[ragged_stop-1]
            aa_seq_nr = full_aa_seq[full_start:full_stop]

        return aa_seq_ragged, aa_seq_nr, exon_prot_start, exon_prot_end, exon_coding_start, exon_coding_end, warnings



def getAllExonSequences(exons, transcripts, genes):
    """
    Run getExonCodingInfo() for all exons in the processed exon dataframe, and add information to exon dataframe

    Parameters
    ----------
    exons: pandas.DataFrame
        Dataframe containing all exon information, created by processedExons() function
    transcripts: pandas.DataFrame
        Dataframe containing all transcript information, created by processedTranscripts() function
    genes: pandas.DataFrame
        DataFrame containing all gene information, created by processedGenes() function

    Returns
    -------
    exons: pandas.DataFrame
        Updated version of exon dataframe containing information about the protein sequence coding for by each exon and its location in the protein sequence
    """
    #initialize lists to be saved in exon
    exon_seqs_ragged = []
    exon_seqs_nr = []
    exon_prot_starts = []
    exon_prot_ends = []
    coding_starts = []
    coding_ends = []
    warnings = []
    logger.info('Getting amino acid sequence coded for by each exon.')
    for e, exon in exons.iterrows():
        #get strand that gene is found on (forward or reverse)
        strand = genes.loc[exon['Gene stable ID'], 'Strand']

        #get protein sequence associated with exon
        results = getExonCodingInfo(exon, transcripts, strand)
        #save data to lists
        exon_seqs_ragged.append(results[0])
        exon_seqs_nr.append(results[1])
        exon_prot_starts.append(results[2])
        exon_prot_ends.append(results[3])
        coding_starts.append(results[4])
        coding_ends.append(results[5])
        warnings.append(results[6])
    
    #save lists to dataframe columns
    exons['Exon Start (Protein)'] = exon_prot_starts
    exons['Exon End (Protein)'] = exon_prot_ends
    exons['Exon AA Seq (Ragged)'] = exon_seqs_ragged
    exons['Exon AA Seq (Full Codon)'] = exon_seqs_nr
    exons['Exon Start (Gene Coding)'] = coding_starts
    exons['Exon End (Gene Coding)'] = coding_ends
    exons['Warnings (AA Seq)'] = warnings
        
    return exons

def exonlength(row, seq_col = 'Exon Sequence', exon_start_col = 'Exon Start (Gene)', exon_end_col = 'Exon End (Gene)'):
    """
    Given a series object with exon information (i.e. row of the exon dataframe), identify the length of the exon based on either the provided sequence or the start and end position of the exon in the gene sequence, if sequence is not in row.

    Parameters
    ----------
    row: pandas.Series
        Row of the exon dataframe or series object with exon information
    seq_col: str
        Name of the column containing the exon sequence. Default is 'Exon Sequence'
    exon_start_col: str
        Name of the column containing the start position of the exon in the gene sequence. Default is 'Exon Start (Gene)'
    exon_end_col: str
        Name of the column containing the end position of the exon in the gene sequence. Default is 'Exon End (Gene)'

    Returns
    ----------
    length: int
        Length of the exon
    """
    exon = row[seq_col]
    if exon is np.nan:
        length = row[exon_start_col]-row[exon_end_col]+1
    else:
        length = len(exon)
        #check if the sequence length matches the length of the exon based on the start and end position of the exon in the gene sequence
        if length != row['Exon End (Gene)']-row['Exon Start (Gene)']+1:
            logger.warning(f'Conflicting exon lengths based on sequence and genomic location for {row["Exon stable ID"]}. Exon sequence length: {length}, Exon length based on genomic location: {row["Exon End (Gene)"]-row["Exon Start (Gene)"]+1}. Using length from exon sequence.')
        
    return length


def translate(row, coding_seq_col = 'Coding Sequence'):
    """
    Given a series object with transcript information (i.e. row of the transcript dataframe), translate the coding sequence into an amino acid sequence, checking to make sure coding sequence is valid: If the coding sequence is not a multiple of 3, does not start with a start codon, or is not available, return np.nan for the amino acid sequence and specify the reason in the warnings list. Function intended for use with pandas apply on transcript dataframe.

    Parameters
    ----------
    row: pandas.Series
        Row of the transcript dataframe or series object with transcript information
    coding_seq_col: str
        Name of the column containing the coding sequence. Default is 'Coding Sequence'

    Returns
    ----------
    row: pandas.Series
        Updated row of the transcript dataframe with a new 'Amino Acid Sequence' column and an updated 'Warnings' column
    """
    #get coding sequence
    seq = row[coding_seq_col]
    
    if seq is np.nan or seq is None:    #check if coding sequence is available
        row['Amino Acid Sequence'] = np.nan
    elif len(seq) % 3 != 0 and seq[0:3] != 'ATG':   #check if sequence starts with start codon and is a multiple of 3
        if row['Warnings'] != row['Warnings']:
            row['Warnings'] = 'Start codon error (not ATG);Partial codon error'
            row['Amino Acid Sequence'] = np.nan
        else:
            row['Warnings'] = row['Warnings'] + ';Start codon error (not ATG);Partial codon error'
            row['Amino Acid Sequence'] = np.nan
    elif len(seq) % 3 != 0:     #check if sequence is a multiple of 3 (no partial codons)
        if row['Warnings'] != row['Warnings']:
            row['Warnings'] = 'Partial codon error'
            row['Amino Acid Sequence'] =np.nan
        else:
            row['Warnings'] = row['Warnings'] + ';Partial codon error'
            row['Amino Acid Sequence'] = np.nan
        #trim sequence explicitly? Currently will not translate
    elif seq[0:3] != 'ATG':     #check if sequence starts with start codon
        if row['Warnings'] != row['Warnings']:
            row['Warnings'] = 'Start codon error (not ATG)'
            row['Amino Acid Sequence'] = np.nan
        else:
            row['Warnings'] = row['Warnings'] + ';Start codon error (not ATG)'
            row['Amino Acid Sequence'] = np.nan
    else:      #if no other errors arise, translate sequence
        row['Warnings'] = np.nan
        #translate and save to row
        coding_strand = Seq(seq)
        aa_seq = str(coding_strand.translate(to_stop = True))
        row['Amino Acid Sequence'] = aa_seq
        
    return row
    


def get_ps_seq(row):
    """
    Given a series object containing a uniprot id associated with a transcript, get the amino acid sequence of the protein from ProteomeScout. Intended for use with pandas apply in getMatchedTranscripts() function

    Parameters
    ----------
    row: pandas.Series
        contains UniProt ID under name 'UniProtKB/Swiss-Prot ID'

    Returns
    ----------
    seq: str
        Amino acid sequence associated with the UniProt ID, obtained from ProteomeScout
    """
    uniprot_id = row['UniProtKB/Swiss-Prot ID']
    seq = config.ps_api.get_sequence(uniprot_id)  # uses base Uniprot ID to get sequence without the isoform info
    return seq
    
def get_uni_id(row):
    """
    Old function, not currently in use or compatible with column names
    """
    if row['Isoform'] == '1':   
        uni_id = row['Uniprot ID']       
    else:
        info = row['Isoform']
        import re
        preans = info.split('[')[1]
        ans = preans.split(']')[0]
        uni_id = re.sub('-', '.', ans)  # this line will be deleted once P.S. is updated, rn the isoform records are listed as PXXX.2 instead of PXXX-2 
        
    return uni_id
    
def get_gencode_seq(seq_align, transcripts):
    """
    Given dataframe containing transcripts of interest and transcript dataframe from processTranscripts(), get the amino acid sequence of the transcripts from the transcript dataframe. 

    Parameters
    ----------
    seq_align: pandas.DataFrame
        dataframe created within getMatchedTranscripts() function containing transcript and uniprot information for canonical uniprot proteins. In theory, this dataframe could be created outside of the function and does not have to be canonical proteins, but it is not currently used elsewhere.
    transcripts: pandas.DataFrame
        Transcript dataframe obtained from processTranscripts()

    Returns
    ----------
    seq: list
        List of amino acid sequences associated with each transcript in seq_align dataframe. If transcript is not found in transcripts dataframe, np.nan is returned.
    """
    seq = []
    for i in seq_align.index:
        gen_id = seq_align.at[i,'Transcript stable ID']
        try:
            seq.append(transcripts.loc[gen_id, 'Amino Acid Sequence'])
        except KeyError:         
            seq.append(np.nan)
        
    return seq
       
def perfect_align(row):
    """
    Given a series object containing the amino acid sequence obtained from Ensembl information and from ProteomeScout, determine if the sequences match exactly. Intended for use with pandas apply in getMatchedTranscripts() function.

    Parameters
    ----------
    row: pandas.Series
        Series object containing the amino acid sequence from Ensembl under the name 'GENCODE Seq' and the amino acid sequence from ProteomeScout under the name 'PS Seq'
    
    Returns
    ----------
    ans: bool
        Indicates if sequences match. True if the sequences match exactly, False if they do not match exactly
    """
    
    u = row['PS Seq']
    v = row['GENCODE Seq']
    ans = u==v
    
    return ans