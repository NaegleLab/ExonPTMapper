import pandas as pd
import numpy as np
import gzip
import re
import sys
import time
from tqdm import tqdm
from ExonPTMapper import config, utility
        
        
def getEventType(gene_cstart, gene_cend, alternative_exons):
    """
    Given the location of the canonical exon which is not present in an alternative transcript, 
    iterate through exons in alternative transcript to
    identify exon that maps similar region of genome. Identify differences between exons (on
    3' or 5' end of exon. If no alternative exons map to the same region of the genome, then this
    is a skipped exon or mixed exon event (labeled 'skipped')
    
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
        - skipped: the genomic region that the canonical exon is found is not used in the alternative transcript. This could indicate either a mixed exon event or a skipped exon
        event
    impacted_region: tuple
        - region of the unspliced gene sequence (found in the gene dataframe created in processing.py) that is impacted by splice event. Provided as tuple: (start of region, stop of region)
    atype: string
        - whether splice event results in a gain or loss of nucleotides. Since a 'skipped' event could potentially be a mixed exon event that could lead to either a gain or loss
        of nucleotides in the transcript, the atype is 'unclear'. Eventually, this should be replaced by separating the skipped and mixed exon events into different categories.
    """
    event = None
    impacted_region = None
    alt_exon = None
    atype = None
    #iterate through exons in alternative transcript
    strand = alternative_exons['Strand'].values[0]
    for j in alternative_exons.index:
        #get location of exon source (in gene)
        gene_astart = int(alternative_exons.loc[j, 'Exon Start (Gene)'])
        gene_aend = int(alternative_exons.loc[j, 'Exon End (Gene)'])
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
            five_prime = gene_cstart == int(alternative_exons.loc[j, 'Exon Start (Gene)'])
            three_prime = gene_cend == int(alternative_exons.loc[j, 'Exon End (Gene)'])
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
            event = 'skipped'
            alt_exon = np.nan
            impacted_region = (gene_cstart,gene_cend)
            atype = 'loss'
            break
        
        #If there is no overlap (based on location in genome) between the canonical exon and the alternative exons, event is skipped
        if event is None:
            event = 'skipped'
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
    if transcript_info['Relative CDS Stop (bp)'] == 'error:no match found':
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


def identifySpliceEvent(canonical_exon, alternative_exons, transcripts = None):
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
        - conserved: ensemble exon id found in both canonical and alternative transcript
        - No Difference: while the same exon id is not present in alternative transcript, there is a different exon id that maps to the same region of the gene (no sequence difference)
        - 3' ASS: the 3' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - 5' ASS: the 5' end of the canonical exon is different in the alternative transcript, but some of the gene is shared
        - skipped: the genomic region that the canonical exon is found is not used in the alternative transcript. This could indicate either a mixed exon event or a skipped exon
        event
    - impacted_region: tuple
        region of the unspliced gene sequence (found in the gene dataframe created in processing.py) that is impacted by splice event. Provided as tuple: (start of region, stop of region)
    - atype: string
        whether splice event results in a gain or loss of nucleotides. Since a 'skipped' event could potentially be a mixed exon event that could lead to either a gain or loss
        of nucleotides in the transcript, the atype is 'unclear'. Eventually, this should be replaced by separating the skipped and mixed exon events into different categories.
    - protein_region: tuple
        region of the protein that is impacted by splice event. Could be fractional if affected region starts/stops in the middle of a codon.
    """
    exon_id = canonical_exon['Exon stable ID']
    strand = canonical_exon['Strand']
    if exon_id not in alternative_exons['Exon stable ID'].values:
        #get location of canonical exon in gene
        gene_cstart = int(canonical_exon['Exon Start (Gene)'])
        gene_cend = int(canonical_exon['Exon End (Gene)'])
        #based on location of exon determine what is the cause of missing exon (skipped, 3'ASS, 5'ASS)
        event, impacted_region, atype, alt_exon = getEventType(gene_cstart, gene_cend, alternative_exons)
        #determine the region of protein (which residues) are affected by splice event
        if event == 'skipped':
            transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon End (Transcript)'])
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
            else:
                protein_region = 'transcript not found'
        elif event == "3' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            #if on the reverse strand, this is actually on the 5' side of the transcript
            if strand == -1:
                event = "5' ASS"
                transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon Start (Transcript)']+region_length)
            else:
                transcript_region = (canonical_exon['Exon End (Transcript)']-region_length, canonical_exon['Exon End (Transcript)'])
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
            else:
                protein_region = 'transcript not found'
        elif event == "5' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            if strand == -1:
                event = "3' ASS"
                transcript_region = transcript_region = (canonical_exon['Exon End (Transcript)']-region_length, canonical_exon['Exon End (Transcript)'])
            else:
                transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon Start (Transcript)']+region_length)
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
            else:
                protein_region = 'transcript not found'
        elif event == "3' and 5' ASS":
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
            protein_region = []
            if canonical_exon['Transcript stable ID'] in transcripts.index.values:
                for reg in transcript_region:
                    protein_region.append(mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], reg))
            else:
                protein_region = 'transcript not found'
        else:
            protein_region = np.nan
    else:
        #if exon id is still in alternative transcript, exon is conserved
        event = 'conserved'
        impacted_region = np.nan
        alt_exon = np.nan
        atype = np.nan
        protein_region = np.nan
            
    return [exon_id, event, alt_exon, impacted_region, atype, protein_region]

def identifySpliceEvents_All(exons, proteins, transcripts, genes, functional = False):
    """
    Given all transcripts in the genome associated with functional proteins + matching protein sequences from ensemble and uniprot, identify all splice events
    that result in a unique exon id or loss of exon id from the canonical transcript (based on transcript associated with canonical uniprot protein).
    
    Parameters
    ----------
    exons: pandas dataframe
        exons dataframe obtained from processing.py
    proteins: pandas dataframe
        proteins dataframe obtained from _______
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
    """
    print('Removing proteins for which the associated gene codes for multiple different proteins')
    #remove multi-protein genes from analysis
    proteins = filterOutMultiProteinGenes(proteins, genes)
    print('Done.\n')
    
    
    print('Removing proteins with no alternative transcripts')
    #split up alternative transcripts into list
    proteins = proteins.dropna(subset = 'Alternative Transcripts (All)')
    proteins['Alternative Transcripts (All)'] = proteins['Alternative Transcripts (All)'].apply(lambda x: x.split(','))
    print('Done.\n')
    
    
    splice_events = []
    #iterate through all proteins in proteins dataframe
    for prot in tqdm(proteins.index, desc = 'Getting splice events for each protein'):
        #identify canonical transcript ID. If multiple, iterate through all
        canonical_trans = proteins.loc[prot, 'Canonical Transcripts']
        if isinstance(canonical_trans, str):
            canonical_trans_list = canonical_trans.split(',')
                #for trans in canonical_trans:
                #    if trans in exons['Transcript stable ID'].values:
                #        canonical_trans = trans
            for canonical_trans in canonical_trans_list:
                #get exons and gene id associated with canonical transcript
                canonical_exons = exons[exons['Transcript stable ID'] == canonical_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                #check to make sure there are exons
                if canonical_exons.shape[0] == 0:
                    continue
                else:
                    gene = canonical_exons['Gene stable ID'].unique()[0]
                    
                    #extract the alternative spliced transcripts (all or only functional)
                    if functional:
                        raise ValueError('filtering based on functionality is currently inactive. Set to functional=False')
                        alt_col = 'Functional Alternatively Spliced Transcripts'
                    else:
                        alt_col = 'Alternative Transcripts (All)'
                    
                    #get alternative transcripts associated with canonical protein
                    alternative_trans = proteins.loc[prot, alt_col]
                    #check to makes sure alternative transcripts exist/we have data for them
                    if len(alternative_trans) != 0 and alternative_trans != 'gene not found':
                        for alt_trans in alternative_trans:
                            #check to make sure individual transcript exists
                            if len(alt_trans) > 0 and alt_trans != 'genenotfound': #####Revisit this, definitely better way to avoid problem where [''] causes errors 
                                #get all exons associated with alternative transcript, ordered by rank in transcript
                                alternative_exons = exons[exons['Transcript stable ID'] == alt_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                                #rare case, but verify that gene_id is same for both alternative exon and canonical exon (alignment needs to match to get results)
                                if alternative_exons.shape[0] == 0:
                                    continue
                                elif alternative_exons['Gene stable ID'].unique()[0] != canonical_exons['Gene stable ID'].unique()[0]:
                                    continue
                                #all other cases, use alignment to obtain the specific splice event for each canonical exon
                                else:
                                    canonical_exon_ID = []
                                    ## Initialize set that holds location of gene
                                    canonical_exon_range_list = []
                                    for index, row in canonical_exons.iterrows():
                                        canonical_exon_range_list = canonical_exon_range_list + list(range(int(row["Exon Start (Gene)"]),int(row["Exon End (Gene)"])))
                                    
                                    canonical_exon_range_list = set(canonical_exon_range_list)
                                    for i in canonical_exons.index:
                                        #get splice event
                                        sevent = identifySpliceEvent(canonical_exons.loc[i], alternative_exons, transcripts)
                                        
                                        #if skipped event, check if any potential MXE's
                                        if sevent[1] == 'skipped':#1. Grab skipped Exon
                                            cgene_start = sevent[3][0]
                                            cgene_end = sevent[3][1]
                                            ## Preprocess and drop any information that does not have the location of the exon 
                                            alternative_exons = alternative_exons[alternative_exons["Exon Start (Gene)"] != 'no match']
                                            # 2. Search for upstream and down stream
                                            alternative_exons["Cgene Start Difference"] =cgene_start - alternative_exons["Exon End (Gene)"].apply(int)
                                            alternative_exons["Cgene End Difference"] = alternative_exons["Exon Start (Gene)"].apply(int) - cgene_end
                                            downstream_exon =  alternative_exons[alternative_exons["Cgene Start Difference"] > 0]
                                            if downstream_exon.shape[0] > 0:
                                                downstream_exon = downstream_exon.loc[[downstream_exon['Cgene Start Difference'].idxmin()]]
                                                                                        #4. Check if there is overlap 
                                                downstream_exon_location = set(list(range(int(downstream_exon["Exon Start (Gene)"]), int(downstream_exon["Exon End (Gene)"]))))
                                                downstream_intersection = len(canonical_exon_range_list.intersection(downstream_exon_location))
                                            else:
                                                downstream_intersection = None
                                            

                                            upstream_exon =  alternative_exons[alternative_exons["Cgene End Difference"] > 0]
                                            if upstream_exon.shape[0] > 0:
                                                upstream_exon = upstream_exon.loc[[upstream_exon['Cgene End Difference'].idxmin()]]
                                                upstream_exon_location = set(list(range(int(upstream_exon["Exon Start (Gene)"]), int(upstream_exon["Exon End (Gene)"]))))
                                                upstream_intersection = len(canonical_exon_range_list.intersection(upstream_exon_location))
                                            else:
                                                upstream_intersection = None
                                            #3. Check if in canonical
                                            sum_of_up_down = sum(downstream_exon["Exon stable ID"].isin(canonical_exons["Exon stable ID"])) + sum(upstream_exon["Exon stable ID"].isin(canonical_exons["Exon stable ID"]))

                                            if sum_of_up_down < 2 and (upstream_intersection == 0 and downstream_intersection == 0):
                                                sevent[1] = 'two MXE'
                                            if sum_of_up_down < 2 and upstream_intersection == 0: 
                                                upstream_end = int(upstream_exon['Exon End (Gene)'])
                                                if not ((canonical_exons['Exon End (Gene)'].astype(int) < upstream_end) & (canonical_exons['Exon End (Gene)'].astype(int) > canonical_exons.loc[i,'Exon End (Gene)'])).any():
                                                    sevent[1] = "mixed"
                                                    sevent[2] = upstream_exon['Exon stable ID'].values[0]
                                                    sevent[4] = 'unclear'
                                            elif sum_of_up_down < 2 and downstream_intersection == 0:
                                                downstream_start = int(downstream_exon['Exon End (Gene)'])
                                                if not ((canonical_exons['Exon Start (Gene)'].astype(int) > downstream_start) & (canonical_exons['Exon Start (Gene)'].astype(int) < canonical_exons.loc[i,'Exon Start (Gene)'])).any():
                                                    sevent[1] = 'mixed'
                                                    sevent[2] = downstream_exon['Exon stable ID'].values[0]
                                                    sevent[4] = 'unclear'
                                                    
                                        #for conserved and no difference exons, check for alterations in protein sequence
                                        elif sevent[1] == 'No Difference' or sevent[1] == 'conserved':
                                            if sevent[1] == 'No Difference':
                                                alt_exon_id = sevent[2]
                                            else:
                                                alt_exon_id = canonical_exons.loc[i, 'Exon stable ID']
                                            alt_exon = alternative_exons[alternative_exons['Exon stable ID'] == alt_exon_id].squeeze()
                                            #check for different amino acid sequence
                                            alt_seq = alt_exon['Exon AA Seq (Full Codon)']
                                            can_seq = canonical_exons.loc[i,'Exon AA Seq (Full Codon)']
                                            if alt_seq == 'No coding seq' or alt_seq == 'Missing Transcript Info' or not isinstance(alt_seq, str):
                                                sevent[1] = 'No Difference, no transcript info on alt'
                                                sevent[4] = 'unclear'
                                            elif can_seq == 'No coding seq' or can_seq == 'Missing Transcript Info' or not isinstance(can_seq, str):
                                                sevent[1] = 'No Difference, no transcript info on can'
                                                sevent[4] = 'unclear'
                                            elif alt_seq == "5' NCR" or alt_seq == "3' NCR":
                                                sevent[1] = 'No Difference, non-coding region'
                                                sevent[4] = 'loss'
                                            elif alt_seq != can_seq:
                                                #check for partial match (different start or stop codon)
                                                if len(alt_seq) > len(can_seq):
                                                    match = re.search(can_seq, alt_seq)
                                                elif len(can_seq) > len(alt_seq):
                                                    match = re.search(alt_seq, can_seq)
                                                    
                                                #if no match exists, check for frameshift or alternative start/stop sites
                                                if match is None:
                                                    #get gene location found in both exons
                                                    shared_loc = list(set(range(int(canonical_exons.loc[i,'Exon Start (Gene)']),int(canonical_exons.loc[i,'Exon End (Gene)']))).intersection(set(range(int(alt_exon['Exon Start (Gene)']),int(alt_exon['Exon End (Gene)'])))))[0]
                                                    #calculate reading frame of canonical, based on shared loc
                                                    can_frame = utility.checkFrame(canonical_exons.loc[i], transcripts.loc[canonical_trans], loc = shared_loc, loc_type = 'Transcript', strand = int(canonical_exons.loc[i, 'Strand']))
                                                    #calcualte reading frame of alternative, based on shared loc
                                                    alt_frame = utility.checkFrame(alt_exon, transcripts.loc[alt_trans], shared_loc, loc_type = 'Transcript', strand = int(alt_exon['Strand']))
                                                    
                                                    if alt_frame != can_frame:
                                                        sevent[1] = 'Frame Shift'
                                                        sevent[4] = 'unclear'
                                                    else:
                                                        sevent[1] = 'Mismatched, but no Frame Shift'
                                                        sevent[4] = 'unclear'
                                                else:
                                                    matched_start = match.span()[0] == 0
                                                    matched_stop = match.span()[1] == len(alt_seq)
                                                    if not matched_start and not matched_stop:
                                                        sevent[1] = 'Alternative Start and Stop Codon'
                                                        sevent[4] = 'unclear'
                                                    elif not matched_start:
                                                        sevent[1] = 'Alternative Start Codon'
                                                        sevent[4] = 'unclear'
                                                    elif not matched_stop:
                                                        sevent[1] = 'Alternative Stop Codon'
                                                        sevent[4] = 'unclear'
                                                    else:
                                                        sevent[1] = 'Partial AA match, but cause unclear'
                                                        sevent[4] = 'unclear'


                                                
                                        #add to array
                                        splice_events.append([prot, gene, canonical_trans, alt_trans]+sevent)
    
    #convert array into pandas dataframe
    splice_events = pd.DataFrame(splice_events, columns = ['Protein', 'Gene','Canonical Transcript', 'Alternative Transcript', 'Exon ID (Canonical)', 'Event Type', 'Exon ID (Alternative)','Genomic Region Affected', 'Loss/Gain', 'Protein Region Affected'])
    return splice_events

def filterOutMultiProteinGenes(proteins, genes):
    multi_genes = genes[genes['Number of Associated Uniprot Proteins'] > 1].index.values
    #convert gene ids to protein ids
    multi_prot = []
    #remove nan values for uniprot id
    trim_translator = config.translator.dropna(subset = 'UniProtKB/Swiss-Prot ID')
    for gene in multi_genes:
        prot_ids = list(trim_translator.loc[trim_translator['Gene stable ID'] == gene, 'UniProtKB/Swiss-Prot ID'].unique())
        if len(prot_ids) > 0:
            #check to make sure protein id is in index (sometimes canonical protein not associated with a transcript and is not in dataframe)
            for ids in prot_ids:
                if ids not in proteins.index.values:
                    prot_ids.remove(ids)
            multi_prot = multi_prot + list(prot_ids)

    proteins = proteins.drop(np.unique(multi_prot))
    return proteins
    


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
            canonical_trans = canonical_trans.split(',')[0]
            
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
                                    event = 'skipped'
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
                                    event = 'skipped'
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