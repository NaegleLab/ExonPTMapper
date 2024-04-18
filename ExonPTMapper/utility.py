import pandas as pd
import gzip
import numpy as np
import warnings
import logging
from Bio import SeqIO
from ExonPTMapper import config


def checkForTranslatorErrors(translator, logger = None):
    """
    Check for potential errors in the translator file (same gene name and transcript ID are associated with multiple protein IDs)

    Parameters
    ----------
    translator: pandas dataframe
        dataframe containing information from the translator file (saved as config.translator when config is imported)
    logger: logging object
        logger object to log warnings to. If None, warnings will only be printed to console

    Returns
    -------
    None, but outputs any warnings to console and log file (if logger is not None)
    """
    #grab entries with duplicate gene ID and transcript ID: potential errors with UniProtID
    test_for_errors = translator[translator.duplicated(['Gene name', 'Gene stable ID', 'Transcript stable ID'], keep = False)]
    
    # see if any errors exist, if so, report
    if test_for_errors.shape[0] > 0:
        error_genes = ', '.join(list(test_for_errors['Gene name'].unique()))
        warnings.warn(f"There are {test_for_errors['Gene name'].nunique()} potential conflicting UniProt IDs in the translator file. Recommend resolving these conflicts manually before continuing, or remove from analysis. See log file for more details.")
        if logger is not None:
            logger.warning(f"There are {test_for_errors['Gene name'].nunique()} potential conflicting UniProt IDs in the translator file for the following genes: {error_genes}. Recommend resolving these conflicts manually before continuing.")

    translator.loc[translator.duplicated(['Gene name', 'Gene stable ID', 'Transcript stable ID'], keep = False), 'Warnings'] = "Conflicting UniProt IDs"

    #test for cases where the dominant UniProt ID is not clear (multiple canonical-seeming entries for a single gene with different UniProt IDs). Exclude genes with potential errors
    test_for_ambiguity = translator[(translator['Uniprot Canonical'] == 'Canonical') & (~translator['Gene name'].isin(list(test_for_errors['Gene name'].unique())))]
    test_for_ambiguity = test_for_ambiguity.groupby('Gene name')['UniProtKB/Swiss-Prot ID'].apply(set).apply(len)
    if test_for_ambiguity.shape[0] > 1:
        ambiguous_genes = ', '.join(list(np.unique(test_for_ambiguity[test_for_ambiguity > 1].index)))
        num_ambigous_genes = len(np.unique(test_for_ambiguity[test_for_ambiguity > 1].index))
        warnings.warn(f"There are {num_ambigous_genes} genes that correspond to multiple UniProt entries. Recommend denoting which UniProt ID should be considered canonical, these genes will be mapped, but will not be considered when assessing splice events")
        if logger is not None:
            logger.warning(f"There are {num_ambigous_genes} genes that correspond to multiple UniProt entries: {ambiguous_genes}. Recommend denoting which UniProt ID should be considered canonical, these genes will be mapped, but will not be considered when assessing splice events")

    #add warning to translator dataframe for genes with multiple possible canonical proteins
    translator.loc[translator['Gene name'].isin(test_for_ambiguity[test_for_ambiguity > 1].index), 'Warnings'] = "Multiple Possible Canonical Proteins for Gene"

#def getIsoformInfo(transcripts):
#    """
#    Get dataframe where each row corresponds to a unique protein isoform, with transcripts with identitical protein information being #in the same row
#
#    Parameters
#    ----------
#    transcripts: pandas dataframe
#    """
#    #add amino acid sequence to translator information
#    merged = config.translator.merge(transcripts['Amino Acid Sequence'], left_on = 'Transcript stable ID', right_index = True)
#    
#    #group information by gene stable ID and amino acid sequence (each unique protein isoform associated with the gene)
#    transcripts =  merged.groupby(['Gene stable ID','Gene name', 'Amino Acid Sequence'])['Transcript stable ID'].apply(set).apply(','.join)
#    proteins = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['UniProtKB/Swiss-Prot ID'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
#    isoform_id = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['UniProtKB isoform ID'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
#    canonical = merged.groupby(['Gene stable ID', 'Gene name', 'Amino Acid Sequence'])['Uniprot Canonical'].apply(set).apply(lambda x: ','.join(y for y in x if y == y))
    
    #combine into dataframe
#    isoforms = pd.concat([proteins, isoform_id, canonical, transcripts], axis = 1).reset_index()
#    isoforms['Uniprot Canonical'] = isoforms['Uniprot Canonical'].replace('', 'Ensembl Alternative')
#    return isoforms


def is_canonical(row):
    """
    Based on the uniprot ID in translator row, label each protein as canonical or isoform. Intended for use with apply function on dataframe

    Parameters
    ----------
    row: pandas series
        row of translator dataframe

    Returns
    -------
    ans: string
        'Canonical' if canonical protein, 'Alternative' if isoform, np.nan if no uniprot ID
    """
    if row['UniProtKB/Swiss-Prot ID'] is np.nan:
        ans = np.nan
    elif row['UniProtKB isoform ID'] is np.nan:
        ans = 'Canonical'
    elif row['UniProtKB isoform ID'].split('-')[1] == '1':
        ans = 'Canonical'
    else:
        ans = 'Alternative'
        
    return ans   


def stringToBoolean(string):
    """
    Convert string object to boolean. False, false, and No all indicate False boolean. True, true, and Yes all indicate True boolean. All other values return np.nan.

    Parameters
    ----------
    string: string
        string to convert to boolean
    
    Returns
    -------
    boolean
    """
    if type(string) == bool:
        return string
    elif string == 'False' or string == 'false' or string == 'No':
        return False
    elif string == 'True' or string == 'true' or string == 'Yes':
        return True
    else:
        return np.nan

def processEnsemblFasta(file, id_col = 'ID', seq_col = 'Seq'):
    """
    Process fasta file downloaded from Ensembl to create dataframe with ID and sequence columns

    Parameters
    ----------
    file: string
        path to fasta file
    id_col: string
        name of column to store ID. Assumes this is found in the .id attribute of SeqIO object
    seq_col: string
        name of column to store sequence. Assumes this is found in the .seq attribute of SeqIO object
    
    Returns
    -------
    pandas dataframe
        dataframe with ID and sequence columns
    """
    data_dict = {id_col:[],seq_col:[]}
    with gzip.open(file,'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            ids = record.id
            seq = str(record.seq)
            
            data_dict[id_col].append(ids)
            data_dict[seq_col].append(seq)
            
    return pd.DataFrame(data_dict)
    
def getUniProtCanonicalIDs(translator, ID_type = 'Transcript'):
    """
    Use translator to extract only the canonical protein ids (either Uniprot IDs or Transcript IDs)

    Parameters
    ----------
    translator: pandas dataframe
        dataframe containing information from the translator file (saved as config.translator when config is imported)
    ID_type: string
        whether to return canonical transcript IDs (Ensembl) or canonical protein IDs (UniProt). Either 'Transcript' or 'Protein'

    Returns
    -------
    list
        list of canonical IDs
    """
    if ID_type == 'Transcript':
        return config.translator.loc[config.translator['canonicals']=='canonical', 'Transcript stable ID'].values
    elif ID_type == 'Protein':
        return translator[translator['canonicals']=='canonical', 'UniProtKB/Swiss-Prot ID'].values
    else:
        print("Please indicate whether you want 'Transcript' or 'Protein' IDs")
        return None

def checkFrame(exon, transcript, loc, loc_type = 'Gene', strand = 1, return_residue = False):
    """
    given location in gene, transcript, or exon, return the location in the frame

    1 = first base pair of codon
    2 = second base pair of codon
    3 = third base pair of codon

    Primary motivation of the function is to determine whether the same gene location in different transcripts is found in the same reading frame

    Parameters
    ----------
    exon: pandas series
        series object containing exon information for exon of interest
    transcript: pandas series
        series object containing transcript information related to exon of interest
    loc: int
        location of nucleotide to check where in frame it is located
    loc_type: string

    """
    if loc_type == 'Gene':
        #calculate location of nucleotide in exon (with 0 being the first base pair of the exon). Consider whether on reverse or forward strand
        if strand == 1:
            loc_in_exon = loc - int(exon['Exon Start (Gene)'])
        else:
            loc_in_exon = exon['Exon End (Gene)'] - loc
        #calculate location of nucleotide in transcript (with 0 being the first base pair of the entire transcript, including UTRs)
        loc_in_transcript = loc_in_exon + int(exon['Exon Start (Transcript)'])
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        coding_loc =(loc_in_transcript - int(transcript['Relative CDS Start (bp)']))
    elif loc_type == 'Exon':
        #calculate location of nucleotide in transcript (with 0 being the first base pair of the entire transcript, including UTRs)
        loc_in_transcript = loc + int(exon['Exon Start (Transcript)'])
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        coding_loc = (loc_in_transcript - int(transcript['Relative CDS Start (bp)']))
    elif loc_type == 'Transcript':
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        coding_loc = (loc - int(transcript['Relative CDS Start (bp)']))
    else:
        print("Invalid loc_type. Can only be based on location in 'Gene','Exon', or 'Transcript'")
        return None
        
    frame = coding_loc % 3 + 1
    #check to make sure coding loc is actually in coding region
    aa_seq = transcript['Amino Acid Sequence']
    if aa_seq != aa_seq:
        frame = np.nan
        if return_residue:
            residue = np.nan
            aa_pos = np.nan
            return frame, residue, aa_pos
        else:
            return frame
    elif coding_loc < 0 or coding_loc/3 >= len(aa_seq):
        frame = -1
        if return_residue:
            residue = "Noncoding"
            aa_pos = np.nan
            return frame, residue, aa_pos
        else:
            return frame
        
    elif return_residue:
        if frame == 1:
            aa_pos = int(coding_loc/3)+1
            residue = transcript['Amino Acid Sequence'][aa_pos-1]
        else:
            aa_pos = np.nan
            residue = np.nan
        return frame, residue, aa_pos
    else:
        return frame

def get_PTMs_PhosphoSitePlus(uniprot_ID, phosphosite):
    """
    Temporary function to bolster data with new phosphositeplus data. Get PTMs for a given Uniprot ID from the PhosphoSitePlus data. You must have created
    the data file from the PhosphoSitePlus using convert_pSiteDataFiles (taken from https://github.com/NaegleLab/CoDIAC/blob/main/CoDIAC/PhosphoSitePlus_Tools.py).

    Parameters
    ----------
    uniprot_ID : str
        Uniprot ID for the protein of interest
    phosphositeplus: str
        Processed phosphositeplus data to extract ptms
    
    Returns
    -------
    PTMs : tuples
        Returns a list of tuples of modifications
        [(position, residue, modification-type),...,]
        
        Returns -1 if unable to find the ID

        Returns [] (empty list) if no modifications  
    
    """
    if uniprot_ID in phosphosite.index:
        mod_str = phosphosite.loc[uniprot_ID, 'modifications']
        if mod_str == 'nan':
            return []
        else:
            mod_list = mod_str.split(';')
            PTMs = []
            for mod in mod_list:
                pos, mod_type = mod.split('-')
                aa = pos[0]
                PTMs.append((aa, pos[1:], mod_type))
            return PTMs
    else:
        print("ERROR: %s not found in PhosphositePlus data"%(uniprot_ID))
        return '-1'

    
    
codon_dict = {'GCA':'A','GCG': 'A','GCC':'A','GCT':'A',
              'TGT':'C','TGC':'C',
              'GAC':'D','GAT':'D',
              'GAA':'E','GAG':'E',
              'TTT':'F','TTC':'F',
              'GGA':'G','GGG':'G','GGC':'G','GGT':'G',
              'CAC':'H','CAT':'H',
              'ATA':'I','ATC':'I','ATT':'I',
              'AAA':'K','AAG':'K',
              'TTG':'L','TTA':'L','CTT':'L','CTC':'L','CTG':'L','CTA':'L',
              'ATG':'M',
              'AAC':'N','AAT':'N',
              'CCA':'P','CCG':'P','CCC':'P','CCT':'P',
              'CAA':'Q','CAG':'Q',
              'AGG':'R','AGA':'R','CGA':'R','CGG':'R','CGC':'R','CGT':'R',
              'TCT':'S','TCC':'S','TCG':'S','TCA':'S','AGC':'S','AGT':'S',
              'ACA':'T','ACG':'T','ACC':'T','ACT':'T',
              'GTA':'V','GTG':'V','GTC':'V','GTT':'V',
              'TGG':'W',
              'TAT':'Y','TAC':'Y',
              'TGA':'*','TAG':'*','TAA':'*'}