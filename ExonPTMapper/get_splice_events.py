import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import logging
import traceback
from Bio import pairwise2
from ExonPTMapper import config, utility
        
#initialize logger
logger = logging.getLogger('SpliceEvents')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(config.processed_data_dir + 'ExonPTMapper.log')
log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(log_format)
logger.addHandler(handler)


def getEventType(gene_cstart, gene_cend, alternative_exons, strand):
    """
    Given the location of the canonical exon which is not present in an alternative transcript, 
    iterate through exons in alternative transcript to
    identify exon that maps similar region of genome. Identify differences between exons (on
    3' or 5' end of exon. If no alternative exons map to the same region of the genome, then this
    is a Skipped exon or mixed exon event (labeled 'Skipped')
    
    Parameters
    ----------
    gene_cstart: int
        location in unspliced gene sequence of canonical exon
    gene_cend: int
        location in unspliced gene sequence of canonical exon
    alternative_exons: pandas dataframe
        subset of exons dataframe obtained from processing.py, which contains all exons associated with alternative transcript
        
    Returns
    -------
    event: string
        type of splice event that explains why canonical exon is not present in alternative transcript (based on ensemble exon IDs)
        - No Difference: while the same exon id is not present in alternative transcript, there is a different exon id that maps to the same region of the gene (no sequence difference)
        - 3' ASS: the 3' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - 5' ASS: the 5' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - Skipped: the genomic region that the canonical exon is found is not used in the alternative transcript. This could indicate either a mixed exon event or a Skipped exon
        event
    impacted_region: tuple
        - region of the unspliced gene sequence (found in the gene dataframe created in processing.py) that is impacted by splice event. Provided as tuple: (start of region, stop of region)
    atype: string
        - whether splice event results in a gain or loss of nucleotides. Since a 'Skipped' event could potentially be a mixed exon event that could lead to either a gain or loss
        of nucleotides in the transcript, the atype is 'unclear'. Eventually, this should be replaced by separating the Skipped and mixed exon events into different categories.
    """
    event = None
    impacted_region = None
    alt_exon = None
    atype = None
    #iterate through exons in alternative transcript
    for j in alternative_exons.index:
        #get location of exon source (in gene)
        gene_astart = alternative_exons.loc[j, 'Exon Start (Gene)']
        gene_aend = alternative_exons.loc[j, 'Exon End (Gene)']
        #determine which regions of canonical exon + alternative exon are shared (if any)
        overlap = set(range(gene_cstart,gene_cend)).intersection(set(range(gene_astart,gene_aend)))
        difference1 =  set(range(gene_cstart,gene_cend)).difference(set(range(gene_astart,gene_aend)))
        difference2 =  set(range(gene_astart,gene_aend)).difference(set(range(gene_cstart,gene_cend)))
        event = None
        #if there is no difference between exons, difference should be 0
        if len(difference1) == 0 and len(difference2) == 0:
            event = 'No Difference'
            impacted_region = np.nan
            alt_exon = alternative_exons.loc[j, 'Exon stable ID']
            atype = np.nan
            break
        #For cases where similarity is not perfect, but there is overlap in exons, determine where the difference occurs
        elif len(overlap) > 0:
            five_prime = gene_cstart == alternative_exons.loc[j, 'Exon Start (Gene)']
            three_prime = gene_cend == alternative_exons.loc[j, 'Exon End (Gene)']
            if not five_prime and not three_prime:
                event = "3' and 5' ASS"
                alt_exon = alternative_exons.loc[j, 'Exon stable ID']
                impacted_region = [(gene_astart,gene_cstart),(gene_cend,gene_aend)]
            elif not five_prime:
                event = "5' ASS"
                alt_exon = alternative_exons.loc[j, 'Exon stable ID']
                if gene_astart < gene_cstart:
                    impacted_region = (gene_astart,gene_cstart)
                    atype = 'gain'
                else:
                    impacted_region = (gene_cstart,gene_astart)
                    atype = 'loss'
            elif not three_prime:
                event = "3' ASS"
                alt_exon = alternative_exons.loc[j, 'Exon stable ID']
                if gene_aend < gene_cend:
                    impacted_region = (gene_aend,gene_cend)
                    atype = 'loss'
                else:
                    impacted_region = (gene_cend,gene_aend)
                    atype = 'gain'
            else:
                event = 'unclear'
            break
        #check to make sure exons haven't passed canonical
        elif (gene_cend < gene_astart and strand == 1) or (gene_cstart > gene_aend and strand == -1):
            event = 'Skipped'
            alt_exon = np.nan
            impacted_region = (gene_cstart,gene_cend)
            atype = 'loss'
            break
        
        #If there is no overlap (based on location in genome) between the canonical exon and the alternative exons, event is Skipped
        if event is None:
            event = 'Skipped'
            alt_exon = np.nan
            impacted_region = (gene_cstart,gene_cend)
            atype = 'loss'
            
    return event, impacted_region, atype, alt_exon

def mapTranscriptToProt(transcript_info, transcript_range):
    """
    Given information about the region of interest within a transcript, identify the associated protein region that it maps to. If part of the transcript region encompasses about
    noncoding region, only include the region that actually encodes for protein.
    
    Parameters
    ----------
    transcript_info: pandas series
        information associated with the transcript of interest. Must contain the following columns: 
        - "CDS Start": location of start of coding region in transcript
        - "CDS Stop": location of stop of the coding region in transcript
        - "coding seq": the full coding sequence in the transcript
    transcript_range: tuple
        indicates the region of interest in the transcript, with the first item being the first nucleotide in the region, and the second item being the second nucleotide in
        the region
        
    Returns
    -------
    protein_region: tuple
        region of protein that is associated with the transcript region. First item in tuple is the first amino acid in the region, second is thelocation of the 
        last amino acid in the region. These could be fractional values if start of transcript is in the middle of a codon
    """
    #can only calculate if coding sequence was mapped to transcript, check to make sure this is true
    if transcript_info['Warnings'] == transcript_info['Warnings']:
        protein_region = 'No coding sequence'
    elif transcript_range[1] > int(transcript_info['Relative CDS Stop (bp)']):       #Check if the end of the transcript region occurs after the end of the coding sequence (and extends into noncoding region)
        if transcript_range[0] > int(transcript_info['Relative CDS Stop (bp)']):     #Check if the start of the transcript region occurs before the end of the coding sequence. If it doesn't, whole region is noncoding
            protein_region = 'non_coding_region'
        else:
            #obtain the protein region based on known location of coding sequence, but stop protein region at the end of coding sequence
            protein_region = ((transcript_range[0] - int(transcript_info['Relative CDS Start (bp)']))/3, len(transcript_info['Coding Sequence'])/3)
    elif transcript_range[0] < int(transcript_info['Relative CDS Start (bp)']):      #Check if the start of the transcript occurs before the start of the coding sequence (and extends into noncoding region)
        if transcript_range[1] < int(transcript_info['Relative CDS Start (bp)']):    #Check if the end of the transcript occurs after the start of the coding sequence. If it doesn't, whole region is noncoding.
            protein_region = 'non_coding_region'
        else:
            #obtain the protein region based on known location of coding sequence, but start protein region at start of coding sequence
            protein_region = (1, (transcript_range[1] - int(transcript_info['Relative CDS Start (bp)']))/3)
    else:
        #obtain the protein region based on known location of coding sequence
        protein_region = ((transcript_range[0] - int(transcript_info['Relative CDS Start (bp)']))/3, (transcript_range[1] - int(transcript_info['Relative CDS Start (bp)']))/3)
    return protein_region


def identifySpliceEvent(canonical_exon, alternative_exons, strand, transcripts = None):
    """
    Given a canonical exon, identify whether it is affected in an alternative transcript, and if it is, the specific alternative splicing event that 
    occurs.
    
    Parameters
    ----------
    canonical_exon: pandas series
        subset of exon dataaframe (obtained from processing.py) which contains information about the exon of interest stemming from a canonical transcript
    alternative_exons: pandas dataframe
        subset of exon dataframe which contains only exons associated with the alternative transcript of interest (alternative exons + canonical exon should come from the 
        same gene)
    transcripts: pandas dataframe
        transcripts dataframe obtained from processing.py, used to identify the protein region impacted by a given splice event. If it is none, the protein region
        is not determined.
        
    Returns
    -------
    List which contains:
    - exon_id: string
        ensemble exon id of the canonical exon
    - event: string
        type of splicing event
        - Conserved: ensemble exon id found in both canonical and alternative transcript
        - No Difference: while the same exon id is not present in alternative transcript, there is a different exon id that maps to the same region of the gene (no sequence difference)
        - 3' ASS: the 3' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - 5' ASS: the 5' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - Skipped: the genomic region that the canonical exon is found is not used in the alternative transcript. This could indicate either a mixed exon event or a Skipped exon
        event
    - impacted_region: tuple
        region of the unspliced gene sequence (found in the gene dataframe created in processing.py) that is impacted by splice event. Provided as tuple: (start of region, stop of region)
    - atype: string
        whether splice event results in a gain or loss of nucleotides. Since a 'Skipped' event could potentially be a mixed exon event that could lead to either a gain or loss
        of nucleotides in the transcript, the atype is 'unclear'. Eventually, this should be replaced by separating the Skipped and mixed exon events into different categories.
    - protein_region: tuple
        region of the protein that is impacted by splice event. Could be fractional if affected region starts/stops in the middle of a codon.
    """
    #get exon id associated with canonical exon
    exon_id = canonical_exon['Exon stable ID']
    #check if exon id is in alternative transcript. If not, determine if what changed in alternative transcript. If it is, it is conserved
    if exon_id not in alternative_exons['Exon stable ID'].values:   
        #get location of canonical exon in gene
        gene_cstart = int(canonical_exon['Exon Start (Gene)'])
        gene_cend = int(canonical_exon['Exon End (Gene)'])

        #based on location of exon determine what is the cause of missing exon (Skipped, 3'ASS, 5'ASS)
        event, impacted_region, atype, alt_exon = getEventType(gene_cstart, gene_cend, alternative_exons, strand)

        #determine the region of protein (which residues) are affected by splice event
        if event == 'Skipped':
            #find the region (base pairs lost) affected by skipped exon event
            transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon End (Transcript)'])

            #as long information is available, find protein region affected 
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
                warnings = np.nan
            else:
                protein_region = np.nan
                warnings = 'Missing Transcript'
        elif event == "3' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            #if on the reverse strand, this is actually on the 5' side of the transcript
            if strand == -1:
                event = "5' ASS"   
                transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon Start (Transcript)']+region_length)
            else:
                transcript_region = (canonical_exon['Exon End (Transcript)']-region_length, canonical_exon['Exon End (Transcript)'])

            #as long information is available, find protein region affected 
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
                warnings = np.nan
            else:
                protein_region = np.nan
                warnings = 'Missing Transcript'
        elif event == "5' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            if strand == -1:
                event = "3' ASS"
                transcript_region = transcript_region = (canonical_exon['Exon End (Transcript)']-region_length, canonical_exon['Exon End (Transcript)'])
            else:
                transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon Start (Transcript)']+region_length)
            
            #as long information is available, find protein region affected 
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
                warnings = np.nan
            else:
                protein_region = np.nan
                warnings = 'Missing Transcript'
        elif event == "3' and 5' ASS":
            #find base pairs affected by alternative splice site
            if strand == 1:
                fiveprime_region_length = impacted_region[0][1]-impacted_region[0][0]   
                threeprime_region_length = impacted_region[1][1]-impacted_region[1][0]
            else:
                fiveprime_region_length = impacted_region[1][1]-impacted_region[1][0]   
                threeprime_region_length = impacted_region[0][1]-impacted_region[0][0]
            transcript_region = [
                (canonical_exon['Exon End (Transcript)'], canonical_exon['Exon End (Transcript)']+fiveprime_region_length),
                (canonical_exon['Exon End (Transcript)']-threeprime_region_length, canonical_exon['Exon End (Transcript)'])
            ]

            #as long information is available, find protein region affected 
            protein_region = []
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                for reg in transcript_region:
                    protein_region.append(mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], reg))
                    warnings = np.nan
            else:
                protein_region = np.nan
                warnings = 'Missing Transcript'
        else:
            protein_region = np.nan
            warnings = np.nan
    else:
        #if exon id is still in alternative transcript, exon is Conserved
        event = 'Conserved'
        impacted_region = np.nan
        alt_exon = np.nan
        atype = np.nan
        protein_region = np.nan
        warnings = np.nan
            
    return [exon_id, event, alt_exon, impacted_region, atype, protein_region, warnings]

def identifySpliceEvents_All(exons, proteins, transcripts, genes, functional = False):
    """
    Given all transcripts in the genome associated with functional proteins + matching protein sequences from ensemble and uniprot, identify all splice events
    that result in a unique exon id or loss of exon id from the canonical transcript (based on transcript associated with canonical uniprot protein).
    
    Parameters
    ----------
    exons: pandas dataframe
        exons dataframe obtained from processing.py
    proteins: pandas dataframe
        proteins dataframe obtained from processing.py
    transcripts: pandas dataframe
        transcripts dataframe obtained from processing.py
    functional: boolean
        indicates whether to only look at splice events resulting in potentially functional proteins (based on TRIFID functional score > 0, obtained from APPRIS database)
        
    Returns
    -------
    splice_events: pandas dataframe
        dataframe containing all observed conservation and splice events, relative to canonical transcript/protein. Each row describes a unique splice event, with columns including:
        - Protein: Uniprot ID of protein associated with splice event
        - Gene: Ensemble Gene ID of protein associated with splice event
        - Canonical Transcript: Ensemble transcript ID of canonical isoform of protein/gene
        - Alternative Transcript: Ensemble transcript ID of alternative isoform of protein/gene associated with splice event
        - Exon ID (Canonical): Ensemble exon ID associated with splice event
        - Event Type: type of splice event obtained from identifySpliceEvents()
        - Genomic Region Affected: start and end of genomic region affected by splice event
        - Loss/Gain: whether splice events result in loss or gain of nucleotides
        - Protein Region Affected: region of canonical protein sequence affected by splice event
        - Warnings: if any errors arose during processing
    """
    logger.info('Getting splice events relative to canonical isoform')

    proteins = proteins[proteins['UniProt Isoform Type'] == 'Canonical']    
    print('Removing proteins for which the associated gene codes for multiple different proteins')
    #remove multi-protein genes from analysis
    proteins = filterOutMultiProteinGenes(proteins, genes)
    print('Done.\n')
    
    
    print('Removing proteins with no alternative transcripts')
    #split up alternative transcripts into list
    original_protein_size = proteins.shape[0]
    proteins = proteins.dropna(subset = 'Variant Transcripts')
    proteins['Variant Transcripts'] = proteins['Variant Transcripts'].apply(lambda x: x.split(';'))
    logger.warning(f'{round(1 - proteins.shape[0]/original_protein_size)*100, 2}% of proteins have no alternative transcripts and were not considered in splice event analysis.')
    print('Done.\n')
    
    print('Removing exons without sequence info')
    exons = exons.dropna(subset = ['Exon Start (Protein)', 'Exon End (Protein)'])
    print('Done.\n')
    

    splice_events = []
    #iterate through all proteins in proteins dataframe
    for prot in tqdm(proteins.index, desc = 'Getting splice events for each protein'):
        #identify canonical transcript ID. If multiple, iterate through all
        canonical_trans = proteins.loc[prot, 'Associated Matched Transcripts']
        if isinstance(canonical_trans, str):
            #grab all transcripts associated with the canonical isoform AND that have information in transcripts dataframe
            canonical_trans_list = [trans for trans in canonical_trans.split(';') if trans in transcripts.index]

            for canonical_trans in canonical_trans_list:
                #get exons and gene id associated with canonical transcript
                canonical_exons = exons[exons['Transcript stable ID'] == canonical_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                #check to make sure there are exons
                if canonical_exons.shape[0] == 0:
                    continue
                else:
                    gene_id = transcripts.loc[canonical_trans, 'Gene stable ID']
                    strand = genes.loc[gene_id, 'Strand']
                    
                    #get alternative transcripts associated with canonical protein
                    alternative_trans = proteins.loc[prot, 'Variant Transcripts']
                    #check to makes sure alternative transcripts exist/we have data for them
                    if len(alternative_trans) != 0:
                        for alt_trans in alternative_trans:
                            #check to make sure individual transcript exists
                            if len(alt_trans) > 0: #####Revisit this, definitely better way to avoid problem where [''] causes errors 

                                #get all exons associated with alternative transcript, ordered by rank in transcript
                                alternative_exons = exons[exons['Transcript stable ID'] == alt_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                                #rare case, but verify that gene_id is same for both alternative exon and canonical exon (alignment needs to match to get results)
                                if alternative_exons.shape[0] == 0:
                                    continue
                                elif transcripts.loc[alt_trans, 'Gene stable ID'] != gene_id:
                                    continue
                                #all other cases, use alignment to obtain the specific splice event for each canonical exon
                                else:
                                    for i in canonical_exons.index:
                                        exon_of_interest = canonical_exons.loc[i]
                                        #check if exon is coding: if not, do not analyze
                                        #get splice event
                                        sevent = identifySpliceEvent(exon_of_interest, alternative_exons, strand, transcripts)
                                        
                                        #if Skipped event, check if any potential MXE's
                                        if sevent[1] == 'Skipped':#1. Grab Skipped Exon
                                            sevent = checkForMXE(sevent, exon_of_interest, canonical_exons, alternative_exons) 
                                                    
                                        #for Conserved and no difference exons, check for alterations in protein sequence
                                        elif sevent[1] == 'No Difference' or sevent[1] == 'Conserved':
                                            if sevent[1] == 'No Difference':
                                                alt_exon_id = sevent[2]
                                            else:
                                                alt_exon_id = canonical_exons.loc[i, 'Exon stable ID']
                                                sevent[2] = alt_exon_id
                                            alt_exon = alternative_exons[alternative_exons['Exon stable ID'] == alt_exon_id].squeeze()
                                            #check for different amino acid sequence
                                            #alt_seq = alt_exon['Exon AA Seq (Full Codon)']
                                            #can_seq = canonical_exons.loc[i,'Exon AA Seq (Full Codon)']

                                            #check if alternative and canonical exons have protein sequence info
                                            alt_seq = alt_exon['Exon AA Seq (Full Codon)']
                                            can_seq = canonical_exons.loc[i,'Exon AA Seq (Full Codon)']
                                            if alt_seq != alt_seq: 
                                                if alt_exon['Warnings (AA Seq)'] == "3' NCR" or alt_exon['Warnings (AA Seq)'] == "5' NCR": #if exon is now in noncoding region, indicate
                                                    sevent[1] = 'No Difference, non-coding region'
                                                    sevent[4] = 'loss'
                                                else:   #otherwise there is missing information of some kind
                                                    sevent[1] = 'No Difference, missing information for alternative transcript'
                                                    sevent[4] = 'unclear'
                                            elif can_seq != can_seq:
                                                if canonical_exons.loc[i,'Warnings (AA Seq)'] == "3' NCR" or canonical_exons.loc[i,'Warnings (AA Seq)'] == "5' NCR": #if exon is now in noncoding region, indicate
                                                    sevent[1] = 'No Difference, non-coding region'
                                                    sevent[4] = 'loss'
                                                else:   #otherwise there is missing information of some kind
                                                    sevent[1] = 'No Difference, missing information for canonical transcript'
                                                    sevent[4] = 'unclear'
                                            else:
                                                alt_seq = alt_exon['Exon AA Seq (Full Codon)']
                                                can_seq = canonical_exons.loc[i,'Exon AA Seq (Full Codon)']
                                                if alt_seq != can_seq:
                                                    #check for partial match (different start or stop codon)
                                                    if len(alt_seq) > len(can_seq):
                                                        match = re.search(can_seq, alt_seq)
                                                    elif len(can_seq) > len(alt_seq):
                                                        match = re.search(alt_seq, can_seq)
                                                        
                                                    #if no match exists, check for frameshift or alternative start/stop sites
                                                    if match is None:
                                                        #get gene location found in both exons
                                                        if strand == 1:
                                                            start_loc = alt_exon['Exon Start (Gene)']
                                                        else:
                                                            start_loc = alt_exon['Exon End (Gene)']
                                                        #calculate reading frame of canonical, based on shared loc
                                                        can_frame = utility.checkFrame(canonical_exons.loc[i], transcripts.loc[canonical_trans], loc = start_loc, loc_type = 'Gene', strand = strand)
                                                        #calcualte reading frame of alternative, based on shared loc
                                                        alt_frame = utility.checkFrame(alt_exon, transcripts.loc[alt_trans], start_loc, loc_type = 'Gene', strand = strand)
                                                        
                                                        if alt_frame != can_frame:
                                                            sevent[1] = 'Frame Shift'
                                                            sevent[4] = 'unclear'
                                                        else:
                                                            sevent[1] = 'Mismatched, but no Frame Shift'
                                                            sevent[4] = 'unclear'
                                                    else:
                                                        matched_start = match.span()[0] == 0
                                                        matched_stop = match.span()[1] == len(alt_seq)
                                                        if not matched_start:
                                                            sevent[1] = 'Alternative Start Codon'
                                                            sevent[4] = 'unclear'
                                                        elif not matched_stop:
                                                            sevent[1] = 'Alternative Stop Codon'
                                                            sevent[4] = 'unclear'
                                                        else:
                                                            sevent[1] = 'Partial AA match, but cause unclear'
                                                            sevent[4] = 'unclear'


                                                
                                        #add to array
                                        splice_events.append([prot, gene_id, canonical_trans, alt_trans]+sevent)
    
    #convert array into pandas dataframe
    splice_events = pd.DataFrame(splice_events, columns = ['Protein', 'Gene','Canonical Transcript', 'Alternative Transcript', 'Exon ID (Canonical)', 'Event Type', 'Exon ID (Alternative)','Genomic Region Affected', 'Loss/Gain', 'Protein Region Affected', 'Warnings'])
    logger.info('Finished getting splice events and saving to processed_data_dir')
    return splice_events

def filterOutMultiProteinGenes(proteins, genes):
    """
    Given a dataframe of proteins and genes, return a trimmed protein dataframe with proteins associated with genes that code for multiple proteins removed. These proteins will be removed in identifySpliceEvents_All().

    Parameters
    ----------
    proteins: pandas dataframe
        proteins dataframe obtained from processing.getProteinInfo()
    genes: pandas dataframe
        genes dataframe obtained from processing.downloadMetaInfo()

    Returns
    -------
    proteins: pandas dataframe
        proteins dataframe with proteins associated with genes that code for multiple proteins removed
    """
    #grab genes with multiple proteins
    multi_genes = genes[genes['Number of Associated Uniprot Proteins'] > 1].index.values
    
    #convert gene ids to protein ids
    multi_prot = []
    #remove nan values for uniprot id
    trim_translator = config.translator[config.translator['UniProt Isoform Type'] == 'Canonical']
    for gene in multi_genes:
        prot_ids = list(trim_translator.loc[trim_translator['Gene stable ID'] == gene, 'UniProtKB isoform ID'].unique())
        if len(prot_ids) > 0:
            #check to make sure protein id is in index (sometimes canonical protein not associated with a transcript and is not in dataframe)
            for ids in prot_ids:
                if ids not in proteins.index.values:
                    prot_ids.remove(ids)
            multi_prot = multi_prot + list(prot_ids)

    logger.warning(f'Removed {len(multi_prot)} proteins from splice event analysis as they are associated with genes with multiple proteins. These include: {", ".join(multi_prot)}')
    proteins = proteins.drop(np.unique(multi_prot))
    return proteins
    
def checkForMXE(sevent, canonical_exon_of_interest, canonical_exons, alternative_exons, max_aa_distance = 20):
    """
    Given a splice event identified as a skipped exon, check if the alterantive transcript has any mutually exclusive exons that are 'replacing' the skipped exon. This is based on 3 criteria defined by Pillman et al. 2013. 

    1. The mutually exclusive exon is the closest exon to the skipped exon in the alternative transcript (i.e. there are not exons located between the skipped exon and the mutually exclusive exon)
    2. The length of the mutually exclusive exon is similar to the skipped exon. By default, this value is set to 20 amino acids, but can be changed by the user.
    3. The reading frame is not altered by the mutually exclusive exon
    4. The sequence of the mutually exclusive exon shares at least 15% sequence similarity to the skipped exon based on alignment scores

    If all of these criteria are met, the skipped exon event is considered a mutually exclusive exon event and the mutual exon is recorded.

    Parameters
    ----------
    sevent: list
        splice event information for skipped exon event, which is obtained from the getSpliceEvent() function
    canonical_exon_of_interest: pandas series
        information about the canonical exon of interest, obtained from the exons dataframe
    canonical_exons: pandas dataframe
        dataframe containing all exons associated with the canonical transcript
    alternative_exons: pandas dataframe
        dataframe containing all exons associated with the alternative transcript
    max_aa_distance: int
        maximum number of amino acids that can be different between the skipped exon and the mutually exclusive exon. If the difference is greater than this value, the skipped exon event is not considered a mutually exclusive exon event

    Returns
    -------
    sevent: list
        updated splice event information for skipped exon event
    """
    #obtain a set consisting of coordinates for all nucleotides in canonical_transcript/exons (to compare to potential MXEs)
    canonical_exons_genomic_locations = []
    for index, row in canonical_exons.iterrows():
        canonical_exons_genomic_locations = canonical_exons_genomic_locations + list(range(int(row["Exon Start (Gene)"]),int(row["Exon End (Gene)"])))
    canonical_exons_genomic_locations = set(canonical_exons_genomic_locations)
    
    #extract location of Skipped exon
    Skipped_exon_start = sevent[3][0]
    Skipped_exon_stop = sevent[3][1]
    
    #find the closest exons in the alternative transcript on either side of the Skipped exon
    downstream_exon, downstream_intersection, upstream_exon, upstream_intersection = findNearbyExons(alternative_exons, canonical_exons_genomic_locations, Skipped_exon_start, Skipped_exon_stop)
    
    
    #check if upstream/downstream exon intersects at all with canonical_transcript exons.
    #If it does not, validate potential MXE with the following criteria
    #1) make sure there are no exons between the Skipped exon and potential MXE in the canonical transcript
    #2) make sure that exon lengths are similar
    #3) make sure reading frame is not altered (same multiple of 3 base pairs)
    #4) make sure sequences are homologous
    if upstream_intersection == 0 and downstream_intersection == 0:
        #first validate upstream exon
        upstream_is_MXE = validatePotentialMXE(canonical_exons, canonical_exon_of_interest, upstream_exon, direction = 'upstream')
        #second validate downstream exon
        downstream_is_MXE = validatePotentialMXE(canonical_exons, canonical_exon_of_interest, upstream_exon, direction = 'downstream')
        
        if upstream_is_MXE and downstream_is_MXE:
            sevent[1] = 'Mutually Exclusive (2)'
            sevent[2] = [upstream_exon['Exon stable ID'], downstream_exon['Exon stable ID']]
            sevent[4] = 'unclear'
        elif upstream_is_MXE:
            sevent[1] = 'Mutually Exclusive'
            sevent[2] = upstream_exon['Exon stable ID']
            sevent[4] = 'unclear'
        elif downstream_is_MXE:
            sevent[1] = 'Mutually Exclusive'
            sevent[2] = downstream_exon['Exon stable ID']
            sevent[4] = 'unclear'
    if upstream_intersection == 0: 
        is_MXE = validatePotentialMXE(canonical_exons, canonical_exon_of_interest, upstream_exon, direction = 'upstream')
        if is_MXE:
            sevent[1] = 'Mutually Exclusive'
            sevent[2] = upstream_exon['Exon stable ID']
            sevent[4] = 'unclear'
    elif downstream_intersection == 0:
        is_MXE = validatePotentialMXE(canonical_exons, canonical_exon_of_interest, downstream_exon, direction = 'downstream')
        if is_MXE:
            sevent[1] = 'Mutually Exclusive'
            sevent[2] = downstream_exon['Exon stable ID']
            sevent[4] = 'unclear'
            
    return sevent
    
           

def findNearbyExons(alternative_exons, canonical_exon_range_list, Skipped_exon_start, Skipped_exon_end):
    """
    Given a skipped exon event, find the closest exons in the alternative transcript on either side of the Skipped exon. This is used to determine if the skipped exon event is actually a mutually exclusive exon event.

    Parameters
    ----------
    alternative_exons: pandas dataframe
        dataframe containing all exons associated with the alternative transcript
    canonical_exon_range_list: list
        list of all genomic locations of canonical exons, to make sure that alternative exons appear in between canonical exons
    Skipped_exon_start: int
        start of skipped exon in genomic coordinates
    Skipped_exon_end: int
        end of skipped exon in genomic coordinates
    
    Returns
    -------
    downstream_exon: pandas series
        information about the closest downstream exon to the skipped exon in the alternative transcript
    downstream_intersection: int
        number of nucleotides in the downstream exon that are also found in the canonical transcript
    upstream_exon: pandas series
        information about the closest upstream exon to the skipped exon in the alternative transcript
    upstream_intersection: int
        number of nucleotides in the upstream exon that are also found in the canonical transcript
    
    """
    ## Preprocess and drop any information that does not have the location of the exon 
    alternative_exons = alternative_exons[alternative_exons["Exon Start (Gene)"] != 'no match']
    
    # Calculate the distance of the each exon in the alternative transcript to the Skipped exon location
    alternative_exons["Cgene Start Difference"] =Skipped_exon_start - alternative_exons["Exon End (Gene)"].apply(int)
    alternative_exons["Cgene End Difference"] = alternative_exons["Exon Start (Gene)"].apply(int) - Skipped_exon_end
    
    #extract alternative exons that are downstream of Skipped exon location (earlier in the gene)
    downstream_exon =  alternative_exons[alternative_exons["Cgene Start Difference"] > 0]
    
    #check if there are any downstream exons at all
    if downstream_exon.shape[0] > 0:
        #find the closest downstream exon
        downstream_exon = downstream_exon.loc[[downstream_exon['Cgene Start Difference'].idxmin()]].squeeze()
        downstream_exon_location = set(list(range(int(downstream_exon["Exon Start (Gene)"]), int(downstream_exon["Exon End (Gene)"]))))
        downstream_intersection = calculate_intersection(canonical_exon_range_list, downstream_exon_location)
    else:
        downstream_intersection = None
    
    #extract alternative exons that are upstream of Skipped exon location (earlier in the gene)
    upstream_exon =  alternative_exons[alternative_exons["Cgene End Difference"] > 0]
    #check if there are any upstream exons at all
    if upstream_exon.shape[0] > 0:
        #find the closest upstream exon
        upstream_exon = upstream_exon.loc[[upstream_exon['Cgene End Difference'].idxmin()]].squeeze()
        upstream_exon_location = set(list(range(int(upstream_exon["Exon Start (Gene)"]), int(upstream_exon["Exon End (Gene)"]))))
        upstream_intersection = calculate_intersection(canonical_exon_range_list, upstream_exon_location)
    else:
        upstream_intersection = None
        
    return downstream_exon, downstream_intersection, upstream_exon, upstream_intersection
    


def calculate_intersection(range1, range2):
    """
    Given two sets (intended to be genomic locations of exons), calculate the number of shared items in each set. If genomic locations, this will be the number of base pairs shared by the two exons.

    Parameters
    ----------
    range1: set
        set of genomic locations (integers) for exon 1
    range2: set
        set of genomic locations (integers) for exon 2
    
    Returns
    -------
    intersection: int
        number of shared items in each set/number of base pairs shared by the two exons
    """
    return len(range1.intersection(range2))
        
def verifyExonAdjacency(alternative_exon, canonical_exons, Skipped_exon, direction = 'upstream'):
    """
    Given an exon is immediately upstream/downstream a skipped exon in the alternative transcript, verify that there are no canonical exons in between the skipped exon and the alternative exon. This is used to determine if the skipped exon event is actually a mutually exclusive exon event.

    Parameters
    ----------
    alternative_exon: pandas series
        information about the alternative exon of interest that is upstream/downstream of the skipped exon
    canonical_exons: pandas dataframe
        dataframe containing all exons associated with the canonical transcript
    Skipped_exon: pandas series
        information about the skipped exon
    direction: str
        direction of alternative exon relative to skipped exon. Can be 'upstream' or 'downstream'
    
    Returns
    -------
    boolean, True if there are no canonical exons in between the skipped exon and the alternative exon (based on genomic coordinates), False otherwise
    """
    if direction == 'downstream':
        downstream_start = int(alternative_exon['Exon End (Gene)'])
        
        #check for canonical exons that are upstream of the downstream exon (same direction as Skipped exon)
        upstream_of_downstream_exon = canonical_exons['Exon Start (Gene)'].astype(int) > downstream_start
        #check for canonical exons that are downstream of the Skipped exon
        downstream_of_Skipped = canonical_exons['Exon Start (Gene)'].astype(int) < Skipped_exon['Exon Start (Gene)']
        
        #if there are any canonical exons that are in between the Skipped exon and the potential MXE, both of the above will be true and it is not an actual MXE
        return ~((upstream_of_downstream_exon) & (downstream_of_Skipped)).any()
        
    elif direction == 'upstream':
        upstream_end = int(alternative_exon['Exon End (Gene)'])
        
        #check for canonical exons that are downstream of the upstream exon (same direction as Skipped exon)
        downstream_of_upstream_exon = canonical_exons['Exon End (Gene)'].astype(int) < upstream_end
        #check for canonical exons that are upstreamstream of the Skipped exon
        upstream_of_Skipped = canonical_exons['Exon End (Gene)'].astype(int) > Skipped_exon['Exon End (Gene)']
        
        #if there are any canonical exons that are in between the Skipped exon and the potential MXE, both of the above will be true and it is not an actual MXE
        return ~((downstream_of_upstream_exon) & (upstream_of_Skipped)).any()
        
def validatePotentialMXE(canonical_exons, Skipped_exon, alternative_exon, direction = 'downstream',
                        max_aa_distance = 20, min_similarity = 0.15):
    """
    Given a skipped exon event, check if a candidate MXE is actually a mutually exclusive exon event. This is based on 5 criteria defined by Pillman et al. 2013.

    1. Mutually exclusive exon is actually coding (not in 5' or 3' NCR)
    2. Mutually exclusive exon is actually adjacent to the skipped exon in the alternative transcript (there aren't any exons in between the skipped exon and the mutually exclusive exon)
    3. Exon shares similar length to skipped exon. By default this assumes MXE is within 20 amino acids of skipped exon, but this can be changed by the user.
    4. MXE does not introduce a frame shift
    5. Sequence of MXE shares at least 15% sequence similarity to skipped exon based on alignment scores

    Parameters
    ----------
    canonical_exons: pandas dataframe
        dataframe containing all exons associated with the canonical transcript
    Skipped_exon: pandas series
        information about the skipped exon
    alternative_exon: pandas series
        information about the alternative exon of interest that is upstream/downstream of the skipped exon (MXE candidate)
    direction: str
        direction of alternative exon relative to skipped exon. Can be 'upstream' or 'downstream'
    max_aa_distance: int
        maximum number of amino acids that can be different between the skipped exon and the mutually exclusive exon. If the difference is greater than this value, the skipped exon event is not considered a mutually exclusive exon event
    min_similarity: float
        minimum percent similarity between the skipped exon and the mutually exclusive exon. If the similarity is less than this value, the skipped exon event is not considered a mutually exclusive exon event. Should be between 0 and 1.
    
    Returns
    -------
    boolean, True if the candidate MXE is actually a mutually exclusive exon event, False otherwise
    """
    #validate candidate MXE is in coding region
    is_coding = (alternative_exon['Exon Start (Protein)'] != "5' NCR") and (alternative_exon['Exon Start (Protein)'] != "3' NCR") and (alternative_exon['Exon Start (Protein)'] != "No coding seq") and (alternative_exon['Exon Start (Protein)'] != "Partial start codon") and (alternative_exon['Exon Start (Protein)'] != "Partial start codon") and (alternative_exon['Exon Start (Protein)'] != "Missing Transcript Info")
    if is_coding:
        #validate Skipped exon and potential mxe in the alternative_exon are actually adjacent
        is_adjacent = verifyExonAdjacency(alternative_exon, canonical_exons, Skipped_exon, direction = direction)
        if is_adjacent:
            #compare exon lengths and make sure they are similar, also make sure frame is the same
            if compareAAlength(Skipped_exon, alternative_exon, max_aa_distance = max_aa_distance):
                #compare exon sequences and check if potential MXE is homologous with Skipped exon
                if compareSeqs(Skipped_exon, alternative_exon, min_similarity = min_similarity):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False
        
def compareAAlength(canonical_exon, alternative_exon, max_aa_distance = 20):
    """
    Given two exons, determine if they are within a similar number of total amino acids, and whether they are found in the same frame. This is used to determine if the skipped exon event is actually a mutually exclusive exon event.

    Parameters
    ----------
    canonical_exon: pandas series
        information about the canonical exon of interest
    alternative_exon: pandas series
        information about the alternative exon of interest that is upstream/downstream of the skipped exon (MXE candidate)
    max_aa_distance: int
        maximum number of amino acids that can be different between the skipped exon and the mutually exclusive exon. If the difference is greater than this value, the skipped exon event is not considered a mutually exclusive exon event
    """
    #get exon aa lengths and check frame of each
    canonical_length = float(canonical_exon['Exon End (Protein)']) - float(canonical_exon['Exon Start (Protein)'])
    canonical_frame = canonical_length % 1
    alternative_length = float(alternative_exon['Exon End (Protein)']) - float(alternative_exon['Exon Start (Protein)'])
    alternative_frame = alternative_length % 1

    #check exon length and frame to make sure they are in reasonable amounts and frame is the same (ends at the same spot)
    if abs(alternative_length - canonical_length) <= max_aa_distance and canonical_frame == alternative_frame:
        return True
    else:
        return False
        
def getNormalizedSimilarity(canonical_exon, alternative_exon):
    """
    Given two exons and their protein sequences, calculate the normalized similarity between the two exons. This is used to determine if the skipped exon event is actually a mutually exclusive exon event.

    Parameters
    ----------
    canonical_exon: pandas series
        information about the canonical exon of interest (skipped exon)
    alternative_exon: pandas series
        information about the alternative exon of interest that is upstream/downstream of the skipped exon (MXE candidate)
    
    Returns
    -------
    normalized_score: float
        normalized similarity score between the two exons. Similarity is normalized relative to the canonical alignment score with itself.
    """
    #get exon sequences
    can_seq = canonical_exon['Exon AA Seq (Full Codon)']
    alt_seq = alternative_exon['Exon AA Seq (Full Codon)']
    
    #make sure both exons have aa sequences and is string
    if not isinstance(can_seq, str) or not isinstance(alt_seq, str):
        return None
        
    
    #align using same gap penalties as Pillmann et al., BMC Bioinformatics 2011
    actual_similarity = pairwise2.align.globalxs(can_seq, alt_seq, -10, -2, score_only = True)
    control_similarity = pairwise2.align.globalxs(can_seq, can_seq, -10, -2, score_only = True)
    normalized_score = actual_similarity/control_similarity

    return normalized_score

def compareSeqs(canonical_exon, alternative_exon, min_similarity = 0.15):
    """
    Given two exons and their protein sequences, determine if the two exons are homologous based on a sequence alignment. This is used to determine if the skipped exon event is actually a mutually exclusive exon event.

    Parameters
    ----------
    canonical_exon: pandas series
        information about the canonical exon of interest (skipped exon)
    alternative_exon: pandas series
        information about the alternative exon of interest that is upstream/downstream of the skipped exon (MXE candidate)
    min_similarity: float
        minimum percent similarity between the skipped exon and the mutually exclusive exon. If the similarity is less than this value, the skipped exon event is not considered a mutually exclusive exon event. Should be between 0 and 1.
    """
    normalized_score = getNormalizedSimilarity(canonical_exon, alternative_exon)
    if normalized_score is None:
        return False
    elif normalized_score >= min_similarity:
        return True
    else:
        return False
    



"""
def identifySpliceEvents(exons, proteins, functional = True):
    splice_events = []
    #iterate through all proteins in proteins dataframe
    for prot in proteins.index:
        #identify canonical transcript ID. If multiple, select the first in list
        canonical_trans = proteins.loc[prot, 'Transcripts']
        if isinstance(canonical_trans, list):
            canonical_trans = canonical_trans[0]
        elif ',' in canonical_trans:
            canonical_trans = canonical_trans.split(';')[0]
            
        #get exons and gene id associated with canonical transcript
        canonical_exons = exons[exons['Transcript stable ID'] == canonical_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
        gene = canonical_exons['Gene stable ID'].unique()[0]
        
        #extract the alternative spliced transcripts (all or only functional)
        if functional:
            alt_col = 'Functional Alternatively Spliced Transcripts'
        else:
            alt_col = 'Alternatively Spliced Transcripts'
        
        alternative_trans = proteins.loc[prot, alt_col]
        if len(alternative_trans) != 0 and alternative_trans != 'gene not found':
            for alt_trans in alternative_trans:
                alternative_exons = exons[exons['Transcript stable ID'] == alt_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                if alternative_exons['Gene stable ID'].unique()[0] != canonical_exons['Gene stable ID'].unique()[0]:
                    event = 'genes do not match'
                    impacted_region = np.nan
                    atype = np.nan
                    protein_region = np.nan
                    splice_events.append([prot, gene, canonical_trans, alt_trans, exon_id, event, impacted_region, atype, protein_region])
                elif 'gene not found' in alternative_exons['Exon Start (Gene)'].unique() or 'gene not found' in canonical_exons['Exon Start (Gene)'].unique() or 'no match' in alternative_exons['Exon Start (Gene)'].unique() or 'no match' in canonical_exons['Exon Start (Gene)'].unique():
                    event = 'gene not found'
                    impacted_region = np.nan
                    atype = np.nan
                    protein_region = np.nan
                    splice_events.append([prot, gene, canonical_trans, alt_trans, exon_id, event, impacted_region, atype, protein_region])
                else:
                    for i in canonical_exons.index:
                        exon_id = canonical_exons.loc[i,'Exon stable ID']
                        if exon_id not in alternative_exons['Exon stable ID'].values:
                            gene_cstart = int(canonical_exons.loc[i, 'Exon Start (Gene)'])
                            gene_cend = int(canonical_exons.loc[i, 'Exon End (Gene)'])
                            for j in alternative_exons.index:
                                gene_astart = int(alternative_exons.loc[j, 'Exon Start (Gene)'])
                                gene_aend = int(alternative_exons.loc[j, 'Exon End (Gene)'])
                                overlap = set(range(gene_cstart,gene_cend)).intersection(set(range(gene_astart,gene_aend)))
                                difference =  set(range(gene_cstart,gene_cend)).difference(set(range(gene_astart,gene_aend)))
                                event = None
                                if len(difference) == 0:
                                    event = 'No Difference'
                                    impacted_region = np.nan
                                    atype = np.nan
                                    protein_region = np.nan
                                    break
                                elif len(overlap) > 0:
                                    five_prime = gene_cstart == int(alternative_exons.loc[j, 'Exon Start (Gene)'])
                                    three_prime = gene_cend == int(alternative_exons.loc[j, 'Exon End (Gene)'])
                                    if not five_prime and not three_prime:
                                        event = "3' and 5' ASS"
                                        impacted_region = [(gene_astart,gene_cstart),(gene_cend, gene_aend)]
                                    elif not five_prime:
                                        event = "5' ASS"
                                        if gene_astart < gene_cstart:
                                            impacted_region = (gene_astart,gene_cstart)
                                            atype = 'gain'
                                        else:
                                            impacted_region = (gene_cstart,gene_astart)
                                            region_length_nc = gene_astart - gene_cstart
                                            transcript_region = (canonical_exons.loc[i, 'Exon Start (Transcript)'], canonical_exons.loc[i, 'Exon Start (Transcript)']+region_length_nc)
                                            if transcripts.loc[canonical_trans, 'CDS Stop'] == 'error:no match found':
                                                protein_region = 'No coding sequence'
                                            elif transcript_region[1] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                                if transcript_region[0] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                                    protein_region = 'non_coding_region'
                                                else:
                                                    protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, len(transcripts.loc[canonical_trans, 'coding seq'])/3)
                                            else:
                                                protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, (transcript_region[1] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3)
                                            atype = 'loss'
                                    elif not three_prime:
                                        event = "3' ASS"
                                        if gene_aend < gene_cend:
                                            impacted_region = (gene_aend, gene_cend)
                                            region_length_nc = gene_cend - gene_aend
                                            transcript_region = (canonical_exons.loc[i, 'Exon End (Transcript)']-region_length_nc, canonical_exons.loc[i, 'Exon End (Transcript)'])
                                            if transcripts.loc[canonical_trans, 'CDS Stop'] == 'error:no match found':
                                                protein_region = 'No coding sequence'
                                            elif transcripts.loc[canonical_trans, 'CDS Stop'] == 'error:no match found':
                                                protein_region = 'No coding sequence'
                                            elif transcript_region[1] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                                if transcript_region[0] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                                    protein_region = 'non_coding_region'
                                                else:
                                                    protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, len(transcripts.loc[canonical_trans, 'coding seq'])/3)
                                            else:
                                                protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, (transcript_region[1] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3)
                                            atype = 'loss'
                                        else:
                                            impacted_region = (gene_cend, gene_aend)
                                            atype = 'gain'
                                    else:
                                        event = 'unclear'
                                    break
                                elif gene_cend < gene_astart:
                                    event = 'Skipped'
                                    impacted_region = (gene_cstart, gene_cend)
                                    atype = 'unclear'
                                    transcript_region = (canonical_exons.loc[i, 'Exon Start (Transcript)'], canonical_exons.loc[i, 'Exon End (Transcript)'])
                                    if transcripts.loc[canonical_trans, 'CDS Stop'] == 'error:no match found':
                                        protein_region = 'No coding sequence'
                                    elif transcript_region[1] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                        if transcript_region[0] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                            protein_region = 'non_coding_region'
                                        else:
                                            protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, len(transcripts.loc[canonical_trans, 'coding seq'])/3)
                                    else:
                                        protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, (transcript_region[1] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3)
                                    break

                                if event is None:
                                    event = 'Skipped'
                                    impacted_region = (gene_cstart, gene_cend)
                                    atype = 'unclear'
                                    transcript_region = (canonical_exons.loc[i, 'Exon Start (Transcript)'], canonical_exons.loc[i, 'Exon End (Transcript)'])
                                    if transcripts.loc[canonical_trans, 'CDS Stop'] == 'error:no match found':
                                        protein_region = 'No coding sequence'
                                    elif transcript_region[1] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                        if transcript_region[0] > int(transcripts.loc[canonical_trans, 'CDS Stop']):
                                            protein_region = 'non_coding_region'
                                        else:
                                            protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, len(transcripts.loc[canonical_trans, 'coding seq'])/3)
                                    else:
                                        protein_region = ((transcript_region[0] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3, (transcript_region[1] - int(transcripts.loc[canonical_trans, 'CDS Start']))/3)



                    splice_events.append([prot, gene, canonical_trans, alt_trans, exon_id, event, impacted_region, atype, protein_region])

    splice_events = pd.DataFrame(splice_events, columns = ['Protein', 'Gene','Canonical Transcript', 'Alternative Transcript', 'Exon ID (Canonical)', 'Event Type', 'Genomic Region Affected', 'Loss/Gain', 'Protein Region Affected'])
    return splice_events"""