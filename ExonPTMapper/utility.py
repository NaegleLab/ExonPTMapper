import pandas as pd
import gzip
import numpy as np
import warnings
import re
from Bio import SeqIO
from Bio.Data import CodonTable
import pybiomart
from ExonPTMapper import config


#packages for web interfacing
import requests
from requests.adapters import HTTPAdapter, Retry
import re


def create_custom_codon_table():
    """
    A small number of proteins contain a rare amino acid selenocysteine (U). An updated codon table is needed to adjust for this possibility, whereby UGA, normally a stop codon, can code for U. I use the custom table functionality of Bio.Seq to update the standard NCBI table

    Returns
    -------
    custom_table: Bio.Data.CodonTable
        Custom codon table with UGA coding for U instead of a stop codon
    """
    # Get the standard codon table
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]

    # Create a copy of the standard table's forward_table (codon to amino acid mapping)
    custom_forward_table = standard_table.forward_table.copy()

    # Replace 'UAG' stop codon with Selenium ('U')
    custom_forward_table['UGA'] = 'T'


    # Create a new codon table with the custom forward_table
    custom_table = CodonTable.CodonTable(
        nucleotide_alphabet=standard_table.nucleotide_alphabet,
        protein_alphabet=standard_table.protein_alphabet + 'U',
        forward_table=custom_forward_table,
        start_codons=standard_table.start_codons,
        stop_codons=["TAA", "TAG"]
    )

    return custom_table


def download_translator(logger, canonical_isoIDs = None):
    """
    Download information from biomart to convert between database IDs
    """
    if canonical_isoIDs is None:
        canonical_isoIDs = config.canonical_isoIDs

    chromosomes = ['X', '20', '1', '6', '3', '7', '12', '11', '4', '17', '2', '16',
       '8', '19', '9', '13', '14', '5', '22', '10', 'Y', '18', '15', '21',
       'MT']
    
    #initialize pybiomart server/dataset
    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',
                   host='http://www.ensembl.org')
    
    #load ID data that relates Ensembl to UniProt
    translator = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id',
                                       'uniprotswissprot', 'uniprot_isoform'],
             filters = {'transcript_biotype':'protein_coding','chromosome_name':chromosomes})

    #indicate whether listed isoforms are canonical, alternative or unannotated in uniprot
    translator["UniProt Isoform Type"] = np.nan
    translator.loc[translator['UniProtKB isoform ID'].isin(canonical_isoIDs.values()), "UniProt Isoform Type"] = 'Canonical' #if annotated as canonical
    translator.loc[(translator['UniProtKB isoform ID'].isna()) & (~translator['UniProtKB/Swiss-Prot ID'].isna()), 'UniProt Isoform Type'] = 'Canonical'  #if the only isoform
    translator.loc[(~translator['UniProtKB isoform ID'].isna()) & (translator["UniProt Isoform Type"].isna()), "UniProt Isoform Type"] = 'Alternative'   #if has isoform ID but is not identified as canonical

    #if isoform id not provided (only one isoform), add '-1' to uniprot id in isoform column
    translator['UniProtKB isoform ID'] = translator.apply(lambda x: x['UniProtKB/Swiss-Prot ID'] + '-1' if x['UniProtKB isoform ID'] != x['UniProtKB isoform ID'] and x['UniProtKB/Swiss-Prot ID'] == x['UniProtKB/Swiss-Prot ID'] else x['UniProtKB isoform ID'], axis = 1)
    

    #identify potential gene name errors and report if any (same gene name and transcript ID are associated with multiple protein IDs)
    translator['Warnings'] = np.nan
    checkForTranslatorErrors(translator, logger=logger)

    #load additional ID data that relates Ensembl to external databases (PDB, RefSeq, CCSD)
    translator2 = dataset.query(attributes=['ensembl_gene_id','external_gene_name', 'ensembl_transcript_id', 'pdb','refseq_mrna', 'ccds'],
             filters = {'transcript_biotype':'protein_coding',
             'chromosome_name':chromosomes})
    
    #merge two dataframes, drop any duplicates caused by merging
    translator = translator.merge(translator2, on = ['Gene stable ID', 'Gene name', 'Transcript stable ID'], how = 'left')
    translator = translator.drop_duplicates()


    return translator



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
    test_for_ambiguity = translator[(translator['UniProt Isoform Type'] == 'Canonical') & (~translator['Gene name'].isin(list(test_for_errors['Gene name'].unique())))]
    test_for_ambiguity = test_for_ambiguity.groupby('Gene name')['UniProtKB/Swiss-Prot ID'].apply(set).apply(len)
    if test_for_ambiguity.shape[0] > 1:
        ambiguous_genes = ', '.join(list(np.unique(test_for_ambiguity[test_for_ambiguity > 1].index)))
        num_ambigous_genes = len(np.unique(test_for_ambiguity[test_for_ambiguity > 1].index))
        warnings.warn(f"There are {num_ambigous_genes} genes that correspond to multiple UniProt entries. Recommend denoting which UniProt ID should be considered canonical, these genes will be mapped, but will not be considered when assessing splice events")
        if logger is not None:
            logger.warning(f"There are {num_ambigous_genes} genes that correspond to multiple UniProt entries: {ambiguous_genes}. Recommend denoting which UniProt ID should be considered canonical, these genes will be mapped, but will not be considered when assessing splice events")

    #add warning to translator dataframe for genes with multiple possible canonical proteins
    translator.loc[translator['Gene name'].isin(test_for_ambiguity[test_for_ambiguity > 1].index), 'Warnings'] = "Multiple Possible Canonical Proteins for Gene"

def create_trim_translator(from_type, to_type):
    """
    Reduce translator object to two database ids of interest, removing any duplicates
    """
    if from_type in config.translator.columns and to_type in config.translator.columns:
        return config.translator[[from_type, to_type]].drop_duplicates()
    elif from_type in config.translator.columns:
        raise ValueError(f'{to_type} is not in the translator object')
    elif to_type in config.translator.columns:
        raise ValueError(f'{from_type} is not in the translator object')
    else:
        raise ValueError(f'Neither {from_type} nor {to_type} are in the translator object')

def convert_IDs(id_to_convert, from_type, to_type):
    """
    Given an id, convert it from one type of database to another using the translator file

    Parameters
    ----------
    id_to_convert: str
        ID to convert
    from_type: str
        column name in translator file to convert from
    to_type: str
        column name in translator file to convert to
    """
    trim_translator = create_trim_translator(from_type, to_type)
    if id_to_convert in trim_translator[from_type].values:
        ids = trim_translator.loc[trim_translator[from_type] == id_to_convert, to_type].values
        if len(ids) > 1:
            return ';'.join(ids)
        else:
            return ids[0]
    else:
        raise ValueError(f'{id_to_convert} not found in translator file')
    
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


#def is_canonical(row):
#    """
#    Based on the uniprot ID in translator row, label each protein as canonical or isoform. Intended for use with apply function on dataframe#
#
#    Parameters
#    ----------
#    row: pandas series
#        row of translator dataframe

#    Returns
#    -------
#    ans: string
#        'Canonical' if canonical protein, 'Alternative' if isoform, np.nan if no uniprot ID
#    """
#    if row['UniProtKB/Swiss-Prot ID'] is np.nan:
#        ans = np.nan
#    elif row['UniProtKB isoform ID'] is np.nan:
#        ans = 'Canonical'
#    elif row['UniProtKB isoform ID'].split('-')[1] == '1':
##        ans = 'Canonical'
#    else:
#        ans = 'Alternative'
#        
#    return ans   


def stringToBoolean(string, null_value = False):
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
    elif string != string:
        return null_value
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
    



def get_PTMs_PhosphoSitePlus(uniprot_ID, phosphosite, isoform_type = 'Canonical'):
    """
    Temporary function to bolster data with new phosphositeplus data. Get PTMs for a given Uniprot ID from the PhosphoSitePlus data. You must have created
    the data file from the PhosphoSitePlus using convert_pSiteDataFiles (taken from https://github.com/NaegleLab/CoDIAC/blob/main/CoDIAC/PhosphoSitePlus_Tools.py).

    Parameters
    ----------
    uniprot_ID : str
        Uniprot ID for the protein of interest
    phosphositeplus: str
        Processed phosphositeplus data to extract ptms
    isoform_type: str
        Type of isoform with PTMs associated with it. If canonical, check for PTMs using the base UniProt ID. Otherwise only use the isoform id
    
    Returns
    -------
    PTMs : tuples
        Returns a list of tuples of modifications
        [(position, residue, modification-type),...,]
        
        Returns -1 if unable to find the ID

        Returns [] (empty list) if no modifications  
    
    """
    #check for uniprot id in index of dataframe
    if uniprot_ID in phosphosite.index:
        mod_str = phosphosite.loc[uniprot_ID, 'modifications']
    elif isoform_type == 'Canonical' and  uniprot_ID.split('-')[0] in phosphosite.index:
        mod_str = phosphosite.loc[uniprot_ID.split('-')[0], 'modifications']
    else:
        return -1

    #check for multiple entries for the same protein (if there are multiple remove)
    if isinstance(mod_str, pd.Series):
        mod_str = mod_str.unique()
        if len(mod_str) > 1:
            raise ValueError("Multiple modifications found for %s"%(uniprot_ID))
        else:
            mod_str = mod_str[0]

    #extract PTMs if any identified
    if mod_str != mod_str:
        return -1
    else:
        mod_list = mod_str.split(';')
        PTMs = []
        for mod in mod_list:
            pos, mod_type = mod.split('-', 1)
            aa = pos[0]
            PTMs.append((pos[1:], aa, mod_type))
        return PTMs

def get_sequence_PhosphoSitePlus(uniprot_ID, phosphosite, isoform_type = 'Canonical'):
    """
    Get the sequence for a given Uniprot ID from the PhosphoSitePlus data. You must have created
    the data file from the PhosphoSitePlus using convert_pSiteDataFiles.

    Parameters
    ----------
    uniprot_ID : str
        Uniprot ID for the protein of interest
    
    Returns
    -------
    sequence : str
        The sequence of the protein of interest. 
        Returns '-1' if sequence not found or if there are duplicate entries
    
    """
    if uniprot_ID in phosphosite.index:
        seq = phosphosite.loc[uniprot_ID, 'sequence']
        if isinstance(seq, str):
            return seq
        else:
            return -1
    elif uniprot_ID.split('-')[0] in phosphosite.index and isoform_type == 'Canonical':
        seq = phosphosite.loc[uniprot_ID.split('-')[0], 'sequence']
        if isinstance(seq, str):
            return seq
        else:
            return -1
    else:
        #print("ERROR: %s not found in PhosphositePlus data"%(uniprot_ID))
        return -1


#UniProt accession services 9adapted from suggested python code on UniProt

def establish_session():
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session, re_next_link

def get_next_link(headers, re_next_link):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url, session, re_next_link):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers, re_next_link)

def get_canonical_isoID(alternative_products, accession):
    """
    Given download of uniprot information containing alternative products/isoforms, identify which isoform is the listed as the canonical isoform. This will be the one that displays the sequence in UniProt
    
    Parameters
    ----------
    alternative_products: str
        alternative_products information returned from UniProtKB
    accession: str
        primary accession number for protein

    Returns
    -------
    canonical_isoid: str
        Isoform ID associated with the canonical isoform of the protein. If only one isoform, isoform ID is listed as '<Accession#>-1'
    """
    #split parts of information by semicolon
    split_list = alternative_products.split(';')
    #check to make sure there are isoforms listed (list will only be 1 length if not)
    if len(split_list) > 1:
        #find index of list with 'Sequence=Displayed'. Canonical id will be just before
        for i, s in enumerate(split_list): 
            if s.strip() == 'Sequence=Displayed':
                displayed_index = i
                break
        #grab canonical isoid, check to make sure it is an isoform id entry
        canonical_isoid = split_list[displayed_index-1]
        if 'IsoId' in canonical_isoid:
            canonical_isoid = canonical_isoid.split('=')[1]
        else:
            raise ValueError(f'Could not find isoform id for {accession}')
    else:
        canonical_isoid = accession + '-1'
    return canonical_isoid

def join_unique_entries(x, sep = ';'):
    """
    For use with groupby, combines all unique entries separated by ';', removing any NaN entries
    """
    #check if only nan entries
    if all(i != i for i in x):
        return np.nan
    if any(sep in i for i in x if i == i): #check if ';' already in entry, if so, split and remove any NaN entries
        split_list = [i.split(sep) for i in x if i == i]
        #split entries in list by ';' and flatten list
        flat_list = [item for sublist in split_list for item in sublist]
        return sep.join(set(flat_list))
    else:
        entry_list = [str(i) for i in x if i == i]
        return sep.join(set(entry_list))

def join_entries(x, sep = ';'):
    #check if only nan entries
    if all(i != i for i in x):
        return np.nan

    if any(sep in i for i in x if i == i): #check if ';' already in entry, if so, split and remove any NaN entries
        split_list = [i.split(sep) for i in x if i == i]
        #split entries in list by ';' and flatten list
        flat_list = [item for sublist in split_list for item in sublist]
        return sep.join(flat_list)

    else:
        entry_list = [str(i) for i in x if i == i]
        return sep.join(entry_list)

def join_except_self(df, group_col, value_col, new_col, sep = ';'):
    """
    For a given dataframe, combines all entries with the same information except for the current row, adds that to the new_col label, and returns the updated dataframe
    
    Parameters
    ----------
    df: pandas DataFrame
        The dataframe to be updated
    group_col: str
        The column to group the dataframe by
    value_col: str
        The column to be combined
    new_col: str
        The new column to be added to the dataframe with the grouped information (excluding the info from the current row)

    Returns
    -------
    df: pandas DataFrame
        updated dataframe with new col labeled with new_col value
    """
    df = df.copy()
    df[new_col] = df.groupby(group_col)[value_col].transform(join_unique_entries, sep)

    #go through each row and remove the value(s) in the new column that is in the value column
    new_values = []
    for i, row in df.iterrows():
        if row[new_col] == row[new_col] and row[value_col] == row[value_col]:
            new_values.append(';'.join([trans for trans in row[new_col].split(sep) if trans not in row[value_col].split(sep)]))
        elif row[value_col] != row[value_col]:
            new_values.append(row[new_col])
        else:
            new_values.append(np.nan)
    df[new_col] = new_values
    return df

    
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