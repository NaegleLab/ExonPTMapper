import pandas as pd
import gzip
from Bio import SeqIO

def processEnsemblFasta(file, id_col = 'ID', seq_col = 'Seq'):
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
    Use translator to extract only the canonical proteins (either Uniprot IDs or Transcript IDs)
    """
    if ID_type == 'Transcript':
        return config.translator.loc[config.translator['canonicals']=='canonical', 'Transcript stable ID'].values
    elif ID_type == 'Protein':
        return translator[translator['canonicals']=='canonical', 'UniProtKB/Swiss-Prot ID'].values
    else:
        print("Please indicate whether you want 'Transcript' or 'Protein' IDs")
        return None

def checkFrame(exon, transcript, loc, loc_type = 'Gene', strand = 1):
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
        frame = (loc_in_transcript - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    elif loc_type == 'Exon':
        #calculate location of nucleotide in transcript (with 0 being the first base pair of the entire transcript, including UTRs)
        loc_in_transcript = loc + int(exon['Exon Start (Transcript)'])
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        frame = (loc_in_transcript - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    elif loc_type == 'Transcript':
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        frame = (loc - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    else:
        print("Invalid loc_type. Can only be based on location in 'Gene','Exon', or 'Transcript'")
    return frame
    
    
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