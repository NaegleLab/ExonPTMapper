import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from ExonPTMapper import config, mapping, utility
import time
import multiprocessing
import itertools

def mapPTMtoAlternative(self, exons, transcripts, canonical_ptm, transcript_id):
    #get alternative exons
    alt_exons = exons[exons['Transcript stable ID'] == transcript_id]
    gene_id = transcripts.loc[transcript_id, 'Gene stable ID']
    
    #get ptm information that matches gene that alternative transcript is found in
    if alt_exons.shape[0] == 0:
        map_result = 'Missing Info'
        new_ptm_position = np.nan
        alt_exon = np.nan
        conserved =np.nan
        can_exon = np.nan
        ragged = np.nan
        canonical_ptm = canonical_ptm.iloc[0]
    else:
        #get ptm information that matches gene that alternative transcript is found in
        canonical_ptm = canonical_ptm[canonical_ptm['Genes'] == gene_id]
        #get gene-specific exon info if more than one associated gene
        if canonical_ptm.shape[0] == 0:
            map_result = 'Missing Info'
            new_ptm_position = np.nan
            alt_exon_id = np.nan
            conserved =np.nan
            canononical_exon_id = np.nan
            ragged = np.nan
        else:
            #grab first canonical transcript in list, should not matter as protein sequence is the same
            canonical_ptm = canonical_ptm.iloc[0]
            gene_loc = canonical_ptm['Gene Location (NC)']
            canonical_exon_id = canonical_ptm['Exon stable ID']
            canonical_trans_id = canonical_ptm['Transcripts']
            canonical_exon = exons[(exons['Exon stable ID'] == canonical_exon_id) & (exons['Transcript stable ID'] == canonical_trans_id)].squeeze()
            
            
        
        
            #check if gene is on forward or reverse strand
            strand = self.genes.loc[self.transcripts.loc[transcript_id, 'Gene stable ID'], 'Strand']
            #find ptm location in alternative transcript (if it exists)
            start = time.time()
            alt_exon_with_ptm = alt_exons[(gene_loc >= alt_exons['Exon Start (Gene)']) & (gene_loc <= alt_exons['Exon End (Gene)'])]
            if alt_exon_with_ptm.shape[0] == 1:
                alt_exon_with_ptm = alt_exon_with_ptm.squeeze()
                #record alternative exon and check if it matches 
                alt_exon_id = alt_exon_with_ptm['Exon stable ID']
                conserved = alt_exon_id == canonical_exon_id
                exon_aa_start = alt_exon_with_ptm['Exon Start (Protein)']

                #check for ragged insertion
                dist_to_boundary = canonical_ptm['Distance to C-terminal Splice Boundary (NC)']
                ragged = dist_to_boundary < 0


                #get location of coding start/end in each exon (in case entire exon does not code for protein). Given as [start,stop]
                canonical_exon = exons[(exons['Exon stable ID'] == canonical_exon_id) & (exons['Transcript stable ID'] == canonical_trans_id)].squeeze()
                can_coding_region = getCodingRegionInGene(canonical_exon,  transcripts.loc[canonical_trans_id], strand)
                alt_coding_region = getCodingRegionInGene(alt_exon_with_ptm, transcripts.loc[transcript_id], strand)
                #check to make sure ptm is in coding region
                if alt_coding_region[0] >= gene_loc or alt_coding_region[1] <= gene_loc:
                    map_result = 'Noncoding region'
                    new_ptm_position = np.nan
                    coding = False
                else:
                    coding = True

                #check to make sure both exons are in the same reading frame
                can_frame = utility.checkFrame(canonical_exon, transcripts.loc[canonical_trans_id], gene_loc, loc_type = 'Gene', strand = strand)
                alt_frame = utility.checkFrame(alt_exon_with_ptm, transcripts.loc[transcript_id], gene_loc, loc_type = 'Gene', strand = strand)

                if can_frame == alt_frame:
                    matched_frame = True
                else:
                    new_ptm_position = np.nan
                    map_result = 'Different Reading Frame'
                    matched_frame = False

                if coding and matched_frame:
                    #check if canonical_ptm and alternative_ptm exist at splice boundary
                    if ragged:
                        splice_boundary = alt_exon_with_ptm['Exon End (Transcript)']
                        transcript_sequence = transcripts.loc[transcript_id, 'Transcript Sequence']
                        #calculate start location of codon in transcript
                        codon_start_in_transcript = splice_boundary - (3+dist_to_boundary)
                        #get new codon in alternative transcript
                        transcript_sequence = transcripts.loc[transcript_id, 'Transcript Sequence']
                        codon = transcript_sequence[codon_start_in_transcript:codon_start_in_transcript+3]
                        new_residue = utility.codon_dict[codon]
                        if new_residue != canonical_ptm['Residue']:
                            new_ptm_position = np.nan
                            map_result = 'Ragged Insertion'
                        else:
                            new_ptm_position = calculatePTMposition(can_coding_region, alt_coding_region, exon_aa_start, canonical_ptm['Exon Location (AA)'], conserved, strand)
                            map_result = 'Success'
                    else:
                        # calculate new ptm position
                        new_ptm_position = calculatePTMposition(can_coding_region, alt_coding_region, exon_aa_start, canonical_ptm['Exon Location (AA)'], conserved, strand)


                        if transcripts.loc[transcript_id, 'Amino Acid Sequence'][new_ptm_position-1] != canonical_ptm['Residue']:
                            map_result = 'Residue Mismatch'
                            new_ptm_position = np.nan
                        else:
                            map_result = 'Success'

                elif ragged:
                    #check for a potential conserved ragged site
                    first_contributing_exon_rank = canonical_ptm['Exon Rank']
                    second_contributing_exon_rank = first_contributing_exon_rank + 1
                    #check if the other contributing exon to ragged site is found in transcript
                    #start_of_other_exon_in_transcript = transcript_loc + (3 + dist_to_boundary)
                    other_contributing_exon = exons[(exons['Transcript stable ID'] == canonical_trans_id) & (exons['Exon rank in transcript'] == second_contributing_exon_rank)].squeeze()
                    loc_other_exon_in_gene = other_contributing_exon['Exon Start (Gene)']
                    alt_exon2 = alt_exons[alt_exons['Exon Start (Gene)'] == loc_other_exon_in_gene]
                    if alt_exon2.shape[0] == 1:
                        alt_exon2 = alt_exon2.squeeze()
                        if strand == 1:
                            #check if the other contributing exon is
                            start_alt_exon_in_transcript = alt_exon2['Exon Start (Transcript)']
                            codon_start = start_alt_exon_in_transcript - (3+dist_to_boundary)
                        else:
                            codon_start = int(alt_exon2['Exon End (Transcript)'])
                        print(codon_start)
                        transcript_sequence = transcripts.loc[transcript_id, 'Transcript Sequence']
                        codon = transcript_sequence[codon_start:codon_start+3]
                        new_residue = utility.codon_dict[codon]
                        if new_residue != canonical_ptm['Residue']:
                            new_ptm_position = np.nan
                            map_result = 'Ragged Insertion'
                        else:
                            new_ptm_position = (codon_start - transcripts.loc[transcript_id, 'Relative CDS Start (bp)'] + 3)/3
                            map_result = 'Success'
                            
                    else:
                        new_ptm_position = np.nan
                        alt_exon_id = np.nan
                        conserved = False
                        map_result = 'Not Found'

                

            else:

                new_ptm_position = np.nan
                alt_exon_id = np.nan
                conserved = False
                map_result = 'Not Found'

    results = [canonical_ptm['Protein'], transcript_id, alt_exon_id, canonical_exon_id, conserved,canonical_ptm['Residue'], canonical_ptm['Modifications'], canonical_ptm['PTM Location (AA)'], new_ptm_position, map_result]
    results = pd.Series(results, index = [ 'Protein','Alternative Transcript', 'Alternative Exon', 'Canonical Exon','Conserved Exon','Residue','Modifications','Canonical Protein Location (AA)', 'Alternative Protein Location (AA)', 'Mapping Result'])
    return results

def calculatePTMposition(can_coding_region, alt_coding_region, exon_aa_start, location_of_ptm_in_exon,conserved, strand):
    if strand == 1:
        if conserved or (can_coding_region[0] == alt_coding_region[0]):
            new_ptm_position = round(location_of_ptm_in_exon+exon_aa_start)

        elif can_coding_region[0] < alt_coding_region[0]:
            start_difference = (alt_coding_region[0] - can_coding_region[0])/3
            new_ptm_position = round(exon_aa_start+location_of_ptm_in_exon-start_difference)
            #print(new_ptm_position, '1')
        else:
            start_difference = (can_coding_region[0] - alt_coding_region[0])/3
            new_ptm_position = round(exon_aa_start+location_of_ptm_in_exon+start_difference)
            #print(new_ptm_position, '2')
    else:
        if conserved or (can_coding_region[1] == alt_coding_region[1]):
            new_ptm_position = round(location_of_ptm_in_exon+exon_aa_start)

        elif can_coding_region[1] < alt_coding_region[1]:
            start_difference = (alt_coding_region[1] - can_coding_region[1])/3
            new_ptm_position = round(exon_aa_start+location_of_ptm_in_exon+start_difference)
            #print(new_ptm_position, '3')
        else:
            start_difference = (can_coding_region[1] - alt_coding_region[1])/3
            new_ptm_position = round(exon_aa_start+location_of_ptm_in_exon-start_difference)

    return new_ptm_position


def getCodingRegionInGene(exon, transcript, strand):
    exon_start = int(transcript['Relative CDS Start (bp)']) - int(exon['Exon Start (Transcript)'])
    exon_stop = int(transcript['Relative CDS Stop (bp)']) - int(exon['Exon Start (Transcript)'])
    #forward strand
    if strand == 1:
        if exon_start < 0:
            gene_start = exon['Exon Start (Gene)']
        else:
            gene_start = exon['Exon Start (Gene)'] + exon_start

        if exon_stop > exon['Exon Length']:
            gene_stop = exon['Exon End (Gene)']
        else:
            gene_stop = exon['Exon Start (Gene)'] + exon_stop
    #reverse strand
    else:
        if exon_start < 0:
            gene_stop = exon['Exon End (Gene)']
        else:
            gene_stop = exon['Exon End (Gene)'] - exon_start
        if exon_stop > exon['Exon Length']:
            gene_start = exon['Exon Start (Gene)']
        else:
            gene_start = exon['Exon End (Gene)'] - exon_stop

    return gene_start, gene_stop


        
def mapBetweenTranscripts_singleProtein(self, exons, transcripts, exploded_ptms, prot_id, PROCESSES = 1):
    alt_trans = self.proteins.loc[prot_id, 'Alternative Transcripts (All)']
    if isinstance(alt_trans, float):
        return None
    else:
        alt_trans = alt_trans.split(',')
        ptms = np.unique(exploded_ptms.loc[exploded_ptms['Protein'] == prot_id,'index'].values)
        alt_ptms = []
        
    
        for trans in alt_trans:
            if trans in transcripts.index.values:
                if PROCESSES > 1:
                    pool = multiprocessing.Pool(processes = PROCESSES)
                    trans_repeats = itertools.repeat(trans)
                    self_repeats = itertools.repeat(self)
                    iterable = zip(self_repeats, ptms, trans_repeats)
                    alt_ptms = pool.starmap(mapPTMtoAlternative, iterable)
                #else:
                for ptm in ptms:
                    canonical_ptm = exploded_ptms[exploded_ptms['index'] == ptm]
                    alt_ptms.append(mapPTMtoAlternative(self, exons, transcripts, canonical_ptm, trans))
        if len(alt_ptms) > 0:
            results = pd.concat(alt_ptms, axis = 1).T   
            results = results[results['Mapping Result'] != 'Missing Info']
            return results
        else:
            return None
			
			


def mapBetweenTranscripts_all(self, results = None, PROCESSES = 1):
    alt_ptms = []
    #restrict to coding information
    trim_exons = self.exons.copy()
    trim_exons['Exon Start (Protein)'] = pd.to_numeric(trim_exons['Exon Start (Protein)'], errors = 'coerce')
    trim_exons['Exon End (Protein)'] = pd.to_numeric(trim_exons['Exon End (Protein)'], errors = 'coerce')
    trim_exons = trim_exons.dropna(subset = ['Exon Start (Protein)', 'Exon End (Protein)'])
    
    #get only transcripts with coding information
    trim_transcripts = self.transcripts.dropna(subset = 'Amino Acid Sequence')
    trim_transcripts = trim_transcripts[trim_transcripts['Relative CDS Start (bp)'] != 'error:no match found']
    trim_transcripts = trim_transcripts[trim_transcripts['Relative CDS Start (bp)'] != 'error:no available transcript sequence']
    trim_transcripts['Relative CDS Start (bp)'] = trim_transcripts['Relative CDS Start (bp)'].astype(int)
    trim_transcripts['Relative CDS Stop (bp)'] = trim_transcripts['Relative CDS Stop (bp)'].astype(int)

    #explode ptm dataframe and process
    exploded_ptms = self.ptm_info.copy()
    exploded_ptms = mapper.explode_PTMinfo()
    exploded_ptms = exploded_ptms[exploded_ptms['Exon stable ID'] != 'Exons Not Found']
    exploded_ptms = exploded_ptms.reset_index()
    exploded_ptms['Gene Location (NC)'] = pd.to_numeric(exploded_ptms['Gene Location (NC)'],errors = 'coerce')
    exploded_ptms = exploded_ptms.dropna(subset = 'Gene Location (NC)')
    exploded_ptms['Exon Location (AA)'] = pd.to_numeric(exploded_ptms['Exon Location (AA)'], errors = 'coerce')
    exploded_ptms = exploded_ptms.dropna(subset = 'Exon Location (AA)')
    exploded_ptms['Distance to C-terminal Splice Boundary (NC)'] = pd.to_numeric(exploded_ptms['Distance to C-terminal Splice Boundary (NC)'], errors = 'coerce')
    exploded_ptms = exploded_ptms.dropna(subset = 'Exon Location (AA)')
    exploded_ptms['Exon Rank'] = pd.to_numeric(exploded_ptms['Exon Rank'], errors = 'coerce')
    exploded_ptms = exploded_ptms.dropna(subset = 'Exon Rank')
    #exploded_ptms = exploded_ptms.merge(self.exons, left_on = ['Exon stable ID', 'Transcripts'], right_on = ['Exon stable ID', 'Transcript stable ID'],how = 'left')
    exploded_ptms = exploded_ptms.drop_duplicates()
    
    
    if results is not None:
        iteration = 0
        for prot_id in tqdm(exploded_ptms['Protein'].unique()):
            try:
                if prot_id not in results['Protein'].unique():
                    prot_alt_ptms = mapBetweenTranscripts_singleProtein(self, trim_exons, trim_transcripts, exploded_ptms, prot_id, PROCESSES = PROCESSES)
                    if prot_alt_ptms is not None:
                        alt_ptms.append(prot_alt_ptms)
                    iteration = iteration + 1
            except:
                e = sys.exc_info()
                print(e) # (Exception Type, Exception Value, TraceBack)
                print(f'{prot_id} is where error occurred')
                return pd.concat([results, pd.concat(alt_ptms)], ignore_index = True)
            if iteration%1000 == 0 and iteration != 0:
                print(f"Iteration {iteration}: saving")
                save_results = pd.concat([results, pd.concat(alt_ptms)], ignore_index = True)
                save_results.to_csv(config.processed_data_dir + 'alt_ptms.csv')
        return pd.concat([results, pd.concat(alt_ptms)], ignore_index = True)
    else:
        iteration = 0
        for prot_id in tqdm(self.ptm_info['Protein'].unique()):
            #try:
            prot_alt_ptms = mapBetweenTranscripts_singleProtein(self, trim_exons, trim_transcripts, exploded_ptms, prot_id, PROCESSES = PROCESSES)
            if prot_alt_ptms is not None:
                alt_ptms.append(prot_alt_ptms)
            #except:
            #    e = sys.exc_info()
            #    print(e) # (Exception Type, Exception Value, TraceBack)
            #    print(f'{prot_id} is where error occurred')
            #    return pd.concat(alt_ptms)
            iteration=iteration + 1
            if iteration%1000 == 0:
                print(f"Iteration {iteration}: saving")
                save_results = pd.concat(alt_ptms)
                save_results.to_csv(config.processed_data_dir + 'alternative_ptms.csv')

        final_results = pd.concat(alt_ptms)
        final_results.to_csv(config.processed_data_dir + 'alternative_ptms.csv')
        return final_results