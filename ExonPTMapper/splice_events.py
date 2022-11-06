import pandas as pd
import numpy as np
import gzip
import re
import sys
import time
from ExonPTMapper import config
        
        
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
    atype = None
    #iterate through exons in alternative transcript
    for j in alternative_exons.index:
        #get location of exon source (in gene)
        gene_astart = int(alternative_exons.loc[j, 'Exon Start (Gene)'])
        gene_aend = int(alternative_exons.loc[j, 'Exon End (Gene)'])
        #determine which regions of canonical exon + alternative exon are shared (if any)
        overlap = set(range(gene_cstart,gene_cend)).intersection(set(range(gene_astart,gene_aend)))
        difference =  set(range(gene_cstart,gene_cend)).difference(set(range(gene_astart,gene_aend)))
        event = None
        #if there is no difference between exons, difference should be 0
        if len(difference) == 0:
            event = 'No Difference'
            impacted_region = np.nan
            atype = np.nan
            break
        #For cases where similarity is not perfect, but there is overlap in exons, determine where the difference occurs
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
                    atype = 'loss'
            elif not three_prime:
                event = "3' ASS"
                if gene_aend < gene_cend:
                    impacted_region = (gene_aend, gene_cend)
                    atype = 'loss'
                else:
                    impacted_region = (gene_cend, gene_aend)
                    atype = 'gain'
            else:
                event = 'unclear'
            break
        #check to make sure exons haven't passed canonical
        elif gene_cend < gene_astart:
            event = 'skipped'
            impacted_region = (gene_cstart, gene_cend)
            atype = 'unclear'
            break
        
        #If there is no overlap (based on location in genome) between the canonical exon and the alternative exons, event is skipped
        if event is None:
            event = 'skipped'
            impacted_region = (gene_cstart, gene_cend)
            atype = 'unclear'
            
    return event, impacted_region, atype

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
    if transcript_info['CDS Stop'] == 'error:no match found':
        protein_region = 'No coding sequence'
    elif transcript_range[1] > int(transcript_info['CDS Stop']):       #Check if the end of the transcript region occurs after the end of the coding sequence (and extends into noncoding region)
        if transcript_range[0] > int(transcript_info['CDS Stop']):     #Check if the start of the transcript region occurs before the end of the coding sequence. If it doesn't, whole region is noncoding
            protein_region = 'non_coding_region'
        else:
            #obtain the protein region based on known location of coding sequence, but stop protein region at the end of coding sequence
            protein_region = ((transcript_range[0] - int(transcript_info['CDS Start']))/3, len(transcript_info['coding seq'])/3)
    elif transcript_range[0] < int(transcript_info['CDS Start']):      #Check if the start of the transcript occurs before the start of the coding sequence (and extends into noncoding region)
        if transcript_range[1] < int(transcript_info['CDS Start']):    #Check if the end of the transcript occurs after the start of the coding sequence. If it doesn't, whole region is noncoding.
            protein_region = 'non_coding_region'
        else:
            #obtain the protein region based on known location of coding sequence, but start protein region at start of coding sequence
            protein_region = (1, (transcript_range[1] - int(transcript_info['CDS Start']))/3)
    else:
        #obtain the protein region based on known location of coding sequence
        protein_region = ((transcript_range[0] - int(transcript_info['CDS Start']))/3, (transcript_range[1] - int(transcript_info['CDS Start']))/3)
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
    if exon_id not in alternative_exons['Exon stable ID'].values:
        #get location of canonical exon in gene
        gene_cstart = int(canonical_exon['Exon Start (Gene)'])
        gene_cend = int(canonical_exon['Exon End (Gene)'])
        #based on location of exon determine what is the cause of missing exon (skipped, 3'ASS, 5'ASS)
        event, impacted_region, atype = getEventType(gene_cstart, gene_cend, alternative_exons)
        #determine the region of protein (which residues) are affected by splice event
        if event == 'skipped':
            transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon End (Transcript)'])
            protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
        elif event == "3' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            transcript_region = (canonical_exon['Exon End (Transcript)']-region_length, canonical_exon['Exon End (Transcript)'])
            protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
        elif event == "5' ASS":
            region_length = impacted_region[1] - impacted_region[0]
            transcript_region = (canonical_exon['Exon Start (Transcript)'], canonical_exon['Exon Start (Transcript)']+region_length)
            protein_region = mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], transcript_region)
        elif event == "3' and 5' ASS":
            fiveprime_region_length = impacted_region[0][1]-impacted_region[0][0]   
            threeprime_region_length = impacted_region[1][1]-impacted_region[1][0]
            transcript_region = [
                (canonical_exon['Exon End (Transcript)'], canonical_exon['Exon End (Transcript)']+fiveprime_region_length),
                (canonical_exon['Exon End (Transcript)']-threeprime_region_length, canonical_exon['Exon End (Transcript)'])
            ]
            protein_region = []
            for reg in transcript_region:
                protein_region.append(mapTranscriptToProt(transcripts.loc[canonical_exon['Transcript stable ID']], reg))
        else:
            protein_region = np.nan
    else:
        #if exon id is still in alternative transcript, exon is conserved
        event = 'conserved'
        impacted_region = np.nan
        atype = np.nan
        protein_region = np.nan
            
    return [exon_id, event, impacted_region, atype, protein_region]

def identifySpliceEvents_All(exons, proteins, transcripts, functional = True):
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
        
        #get alternative transcripts associated with canonical protein
        alternative_trans = proteins.loc[prot, alt_col]
        #check to makes sure alternative transcripts exist/we have data for them
        if len(alternative_trans) != 0 and alternative_trans != 'gene not found':
            for alt_trans in alternative_trans:
                #check to make sure individual transcript exists
                if len(alternative_trans[0]) > 0 and alternative_trans[0] != 'genenotfound': #####Revisit this, definitely better way to avoid problem where [''] causes errors 
                    #get all exons associated with alternative transcript, ordered by rank in transcript
                    alternative_exons = exons[exons['Transcript stable ID'] == alt_trans].sort_values(by = 'Exon rank in transcript', ascending = True)
                    #rare case, but verify that gene_id is same for both alternative exon and canonical exon (alignment needs to match to get results)
                    if alternative_exons['Gene stable ID'].unique()[0] != canonical_exons['Gene stable ID'].unique()[0]:
                        exon_id = np.nan
                        event = 'genes do not match'
                        impacted_region = np.nan
                        atype = np.nan
                        protein_region = np.nan
                        splice_events.append([prot, gene, canonical_trans, alt_trans, exon_id, event, impacted_region, atype, protein_region])
                    #another rare case, but check to make sure a gene is associated with alternative exon (again, need alignment to get splice event)
                    elif 'gene not found' in alternative_exons['Exon Start (Gene)'].unique() or 'gene not found' in canonical_exons['Exon Start (Gene)'].unique() or 'no match' in alternative_exons['Exon Start (Gene)'].unique() or 'no match' in canonical_exons['Exon Start (Gene)'].unique():
                        exon_id = np.nan
                        event = 'gene not found'
                        impacted_region = np.nan
                        atype = np.nan
                        protein_region = np.nan
                        splice_events.append([prot, gene, canonical_trans, alt_trans, exon_id, event, impacted_region, atype, protein_region])
                    #all other cases, use alignment to obtain the specific splice event for each canonical exon
                    else:
                        for i in canonical_exons.index:
                            #get splice event
                            sevent = identifySpliceEvent(canonical_exons.loc[i], alternative_exons, transcripts)
                            #add to array
                            splice_events.append([prot, gene, canonical_trans, alt_trans]+sevent)
    
    #convert array into pandas dataframe
    splice_events = pd.DataFrame(splice_events, columns = ['Protein', 'Gene','Canonical Transcript', 'Alternative Transcript', 'Exon ID (Canonical)', 'Event Type', 'Genomic Region Affected', 'Loss/Gain', 'Protein Region Affected'])
    return splice_events


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