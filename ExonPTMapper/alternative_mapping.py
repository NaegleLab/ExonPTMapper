import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from ExonPTMapper import config, mapping, utility
import time
import multiprocessing
import itertools

def mapPTMtoAlternative(mapper, exons, transcripts, canonical_ptm, transcript_id):
    #get alternative exons
    alt_exons = exons[exons['Transcript stable ID'] == transcript_id]
    gene_id = transcripts.loc[transcript_id, 'Gene stable ID']
    
    #get ptm information that matches gene that alternative transcript is found in
    if alt_exons.shape[0] == 0:
        map_result = 'Missing Info'
        new_ptm_position = np.nan
        alt_exon_id = np.nan
        conserved =np.nan
        canonical_exon_id = np.nan
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
            canonical_exon_id = np.nan
            ragged = np.nan
        else:
            #grab first canonical transcript in list, should not matter as protein sequence is the same
            canonical_ptm = canonical_ptm.iloc[0]
            gene_loc = canonical_ptm['Gene Location (NC)']
            canonical_exon_id = canonical_ptm['Exon stable ID']
            canonical_trans_id = canonical_ptm['Transcripts']
            canonical_exon = exons[(exons['Exon stable ID'] == canonical_exon_id) & (exons['Transcript stable ID'] == canonical_trans_id)].squeeze()
            
            #check for ragged insertion
            dist_to_boundary = canonical_ptm['Distance to C-terminal Splice Boundary (NC)']
            ragged = dist_to_boundary < 0
        
        
            #check if gene is on forward or reverse strand
            strand = mapper.genes.loc[mapper.transcripts.loc[transcript_id, 'Gene stable ID'], 'Strand']
            #find ptm location in alternative transcript (if it exists)
            start = time.time()
            alt_exon_with_ptm = alt_exons[(gene_loc >= alt_exons['Exon Start (Gene)']) & (gene_loc <= alt_exons['Exon End (Gene)'])]
            if alt_exon_with_ptm.shape[0] == 1:
                alt_exon_with_ptm = alt_exon_with_ptm.squeeze()
                #record alternative exon and check if it matches 
                alt_exon_id = alt_exon_with_ptm['Exon stable ID']
                conserved = alt_exon_id == canonical_exon_id
                exon_aa_start = alt_exon_with_ptm['Exon Start (Protein)']


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

                    # calculate new ptm position
                    new_ptm_position = calculatePTMposition(can_coding_region, alt_coding_region, exon_aa_start, canonical_ptm['Exon Location (AA)'], conserved, strand)


                    if transcripts.loc[transcript_id, 'Amino Acid Sequence'][new_ptm_position-1] != canonical_ptm['Residue']:
                        if ragged:
                            map_result = 'Ragged Insertion'
                            new_ptm_position = np.nan
                        else:
                            map_result = 'Residue Mismatch'
                            new_ptm_position = np.nan
                    else:
                        map_result = 'Success'

            elif ragged:
                conserved = False
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
                    alt_exon_id = alt_exon2['Exon stable ID']
                    if strand == 1:
                        #check if the other contributing exon is
                        start_alt_exon_in_transcript = alt_exon2['Exon Start (Transcript)']
                        codon_start = start_alt_exon_in_transcript - (3+dist_to_boundary)
                    else:
                        codon_start = int(alt_exon2['Exon End (Transcript)'])

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


        
def mapBetweenTranscripts_singleProtein(mapper, exons, transcripts, exploded_ptms, prot_id, PROCESSES = 1):
    alt_trans = mapper.proteins.loc[prot_id, 'Alternative Transcripts (All)']
    if isinstance(alt_trans, float):
        return None
    else:
        alt_trans = alt_trans.split(',')
        ptms = np.unique(exploded_ptms.loc[exploded_ptms['Protein'] == prot_id,'PTM'].values)
        alt_ptms = []
        
    
        for trans in alt_trans:
            if trans in transcripts.index.values:
                if PROCESSES > 1:
                    pool = multiprocessing.Pool(processes = PROCESSES)
                    trans_repeats = itertools.repeat(trans)
                    mapper_repeats = itertools.repeat(mapper)
                    iterable = zip(mapper_repeats, ptms, trans_repeats)
                    alt_ptms = pool.starmap(mapPTMtoAlternative, iterable)
                #else:
                for ptm in ptms:
                    canonical_ptm = exploded_ptms[exploded_ptms['PTM'] == ptm]
                    alt_ptms.append(mapPTMtoAlternative(mapper, exons, transcripts, canonical_ptm, trans))
        if len(alt_ptms) > 0:
            results = pd.concat(alt_ptms, axis = 1).T   
            results = results[results['Mapping Result'] != 'Missing Info']
            return results
        else:
            return None
			
			


def mapBetweenTranscripts_all(mapper, results = None, PROCESSES = 1):
    alt_ptms = []
    #restrict to coding information
    trim_exons = mapper.exons.copy()
    trim_exons['Exon Start (Protein)'] = pd.to_numeric(trim_exons['Exon Start (Protein)'], errors = 'coerce')
    trim_exons['Exon End (Protein)'] = pd.to_numeric(trim_exons['Exon End (Protein)'], errors = 'coerce')
    trim_exons = trim_exons.dropna(subset = ['Exon Start (Protein)', 'Exon End (Protein)'])
    
    #get only transcripts with coding information
    trim_transcripts = mapper.transcripts.dropna(subset = 'Amino Acid Sequence')
    trim_transcripts = trim_transcripts[trim_transcripts['Relative CDS Start (bp)'] != 'error:no match found']
    trim_transcripts = trim_transcripts[trim_transcripts['Relative CDS Start (bp)'] != 'error:no available transcript sequence']
    trim_transcripts['Relative CDS Start (bp)'] = trim_transcripts['Relative CDS Start (bp)'].astype(int)
    trim_transcripts['Relative CDS Stop (bp)'] = trim_transcripts['Relative CDS Stop (bp)'].astype(int)

    #explode ptm dataframe and process
    exploded_ptms = mapper.ptm_info.copy()
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
    #exploded_ptms = exploded_ptms.merge(mapper.exons, left_on = ['Exon stable ID', 'Transcripts'], right_on = ['Exon stable ID', 'Transcript stable ID'],how = 'left')
    exploded_ptms = exploded_ptms.drop_duplicates()
    
    
    if results is not None:
        iteration = 0
        for prot_id in tqdm(exploded_ptms['Protein'].unique()):
            try:
                if prot_id not in results['Protein'].unique():
                    prot_alt_ptms = mapBetweenTranscripts_singleProtein(mapper, trim_exons, trim_transcripts, exploded_ptms, prot_id, PROCESSES = PROCESSES)
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
        for prot_id in tqdm(mapper.ptm_info['Protein'].unique()):
            #try:
            prot_alt_ptms = mapBetweenTranscripts_singleProtein(mapper, trim_exons, trim_transcripts, exploded_ptms, prot_id, PROCESSES = PROCESSES)
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
        
        
def mapPTMtoExons(mapper, ptm, trim_exons = None, alternative_only = True):
    #check to make sure gene location is str so that multiple locs can be split into list
    if not isinstance(ptm['Gene Location (NC)'], str):
        ptm['Gene Location (NC)'] = str(ptm['Gene Location (NC)'])
        
    #reduce exons dataframe to exons associated with transcripts with available information, if not provided
    if trim_exons is None:
        #identify transcripts (plus associated exons) with available transcript and amino acid sequence
        available_transcripts = mapper.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
        #if desired, isolate only alternative transcripts
        if alternative_only:
            alternative_transcripts = config.translator.loc[config.translator['Uniprot Canonical'] != 'Canonical', 'Transcript stable ID']
            available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
            
        trim_exons = mapper.exons[mapper.exons['Transcript stable ID'].isin(available_transcripts)]
        #extract only exon information required for mapping, making sure no duplicates
        trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()
    
    results = None
    #isolate info into unique genes (if multiple associated with protein/ptm of interest)
    gene_locs = ptm['Gene Location (NC)'].split(',')
    exons = ptm['Exon stable ID'].split(',')
    genes = mapper.genes.loc[ptm['Genes'].split(',')]
    #for each gene information, identify exons with conserved ptm site, based on genome location of ptm
    for g in range(len(gene_locs)):
        gene = genes.iloc[g]
        source_exon_id = exons[g]
        gene_loc = gene_locs[g]
        if gene_loc != 'Exons Not Found' and gene_loc != 'CDS fail':
            gene_loc = int(gene_loc)
            strand = gene['Strand']
            #grab all exons containing ptm site (based on genome location and ensuring they come from same gene info)
            ptm_exons = trim_exons[(trim_exons['Exon Start (Gene)'] <= gene_loc) & (trim_exons['Exon End (Gene)'] >= gene_loc) & (trim_exons['Gene stable ID'] == gene.name)].copy()
            if ptm_exons.shape[0] != 0:
                ptm_exons['PTM'] = ptm.name
                ptm_exons['Source Exon'] = source_exon_id
                chromosome = gene['Chromosome/scaffold name']

                #check if ptm is at the boundary (ragged site)
                distance_to_bound = ptm_exons.apply(getDistanceToBoundary, args = (gene_loc, strand), axis = 1)
                ptm_exons['Ragged'] = distance_to_bound <= 1
                #identify location of ptm in new exons (different equations depending on if gene is on the forward or reverse strand)
                if not ptm_exons['Ragged'].any():
                    ptm_exons['Genomic Coordinates'] = getGenomicCoordinates(chromosome, gene_loc, strand)
                    ptm_exons['Second Exon'] = np.nan
                    #check if frame matches for given event: if it does, identify new residue + position in protein associated with exon
                    ptm_loc_info = ptm_exons.apply(getPTMLoc, args = (mapper, gene_loc, strand), axis = 1)
                    #unpack results and save to dataframe
                    ptm_exons['Frame'] = [x[0] for x in ptm_loc_info]
                    ptm_exons['Residue'] = [x[1] for x in ptm_loc_info]
                    ptm_exons['Position'] = [x[2] for x in ptm_loc_info]
                else:
                    coords = []
                    aa_pos_list = []
                    residue_list = []
                    frame_list = []
                    second_exon_list = []
                    for i, row in ptm_exons.iterrows():
                        if row['Ragged']:
                            next_rank = row['Exon rank in transcript'] + 1
                            transcript = mapper.exons[mapper.exons['Transcript stable ID'] == row['Transcript stable ID']]
                            next_exon = transcript[transcript['Exon rank in transcript'] == next_rank].squeeze()
                            ragged_loc = next_exon['Exon Start (Gene)'] if strand == 1 else next_exon['Exon End (Gene)']
                            if isinstance(ragged_loc, pd.Series):
                                coords.append(np.nan)
                                frame_list.append(np.nan)
                                residue_list.append(np.nan)
                                aa_pos_list.append(np.nan)
                                second_exon_list.append(np.nan)
                            else:
                                #check if frame matches for given event: if it does, identify new residue + position in protein associated with exon
                                frame, residue, aa_pos = getPTMLoc(row, mapper, gene_loc, strand)
                                frame_list.append(frame)
                                residue_list.append(residue_list)
                                aa_pos_list.append(aa_pos_list)
                                second_exon_list.append(next_exon['Exon stable ID'])
                                #get genomic coordinates (from both contributing exons)
                                coords.append(getRaggedCoordinates(chromosome, gene_loc, ragged_loc, distance_to_bound[i], strand))

                        else:
                            coords.append(getGenomicCoordinates(chromosome, gene_loc, strand))
                            #check if frame matches for given event: if it does, identify new residue + position in protein associated with exon
                            frame, residue, aa_pos = getPTMLoc(row, mapper, gene_loc,strand)
                            frame_list.append(frame)
                            residue_list.append(residue_list)
                            aa_pos_list.append(aa_pos_list)
                            second_exon_list.append(np.nan)

                    ptm_exons['Genomic Coordinates'] = coords
                    ptm_exons['Frame'] = frame_list
                    ptm_exons['Residue'] = residue_list
                    ptm_exons['Position'] = aa_pos_list
                    ptm_exons['Second Exon'] = second_exon_list
                if results is None:
                    results = ptm_exons.copy()
                else:
                    results = pd.concat([results, ptm_exons])
                    
    if results is None:
        return None
    else:
        return results
          
def mapPTMstoAlternativeExons(mapper):
    #get all alternative transcripts with available coding info
    available_transcripts = mapper.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
    alternative_transcripts = config.translator.loc[config.translator['Uniprot Canonical'] != 'Canonical', 'Transcript stable ID']
    available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
    
    #grab exons associated with available transcripts
    trim_exons = mapper.exons[mapper.exons['Transcript stable ID'].isin(available_transcripts)]
    trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()
    
    #convert gene location into string
    mapper.ptm_info['Gene Location (NC)'] = mapper.ptm_info['Gene Location (NC)'].astype(str)
    
    alt_ptms = None
    for i,ptm in tqdm(mapper.ptm_info.iterrows()):
        ptm_exons = mapPTMtoExons(mapper, ptm, trim_exons = trim_exons)
        if alt_ptms is None:
            alt_ptms = ptm_exons.copy()
        else:
            alt_ptms = pd.concat([alt_ptms, ptm_exons])
    
    #remove residual exon columns that are not needed
    alt_ptms = alt_ptms.drop(['Exon End (Gene)', 'Exon Start (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)', 'Exon rank in transcript'], axis = 1)
    #grab ptm specific info
    #alt_ptms['Protein'] = alt_ptms['PTM'].apply(lambda x: x.split('_')[0])
    #alt_ptms['Canonical Residue'] = alt_ptms['PTM'].apply(lambda x: x.split('_')[1][0])
    #alt_ptms['Canonical Protein Location (AA)'] = alt_ptms['PTM'].apply(lambda x: x.split('_')[1][1:])
          
    #add ptms that were unsuccessfully mapped to alternative transcripts
    alternative = mapper.proteins.dropna(subset = 'Alternative Transcripts (All)').copy()
    alternative['Alternative Transcripts (All)'] = alternative['Alternative Transcripts (All)'].apply(lambda x: x.split(','))
    alternative = alternative.explode('Alternative Transcripts (All)').reset_index()
    alternative = alternative[['UniProtKB/Swiss-Prot ID', 'Alternative Transcripts (All)']]
    alternative = alternative[alternative['Alternative Transcripts (All)'].isin(available_transcripts)]
    alternative = alternative.rename({'Alternative Transcripts (All)': 'Transcript stable ID', 'UniProtKB/Swiss-Prot ID':'Protein'}, axis = 1)
    ptms = mapper.ptm_info.reset_index()[['index', 'Protein']].drop_duplicates()
    ptms = ptms.rename({'index':'PTM'}, axis = 1)
    alternative = alternative.merge(ptms, on = 'Protein')
    #identify PTMs that were found in alternative transcripts
    #missing = ~alternative[['Transcript stable ID', 'PTM']].isin(alt_ptms[['Transcript stable ID', 'PTM']])
    #missing_alternative = alternative[missing]

    
    alt_ptms = alt_ptms.merge(alternative, on = ['Transcript stable ID', 'PTM'], how = 'outer')
    alt_ptms['Canonical Residue'] = alt_ptms['PTM'].apply(lambda x: x.split('_')[1][0])
    alt_ptms['Canonical Protein Location (AA)'] = alt_ptms['PTM'].apply(lambda x: x.split('_')[1][1:])
   
    
    #rename columns
    alt_ptms = alt_ptms.rename({'Exon stable ID': 'Exon ID (Alternative)', 'Source Exon': 'Exon ID (Canonical)', 'Transcript stable ID': 'Alternative Transcript'}, axis = 1)
    
    #identify cases where ptms were successfully or unsuccessfully mapped
    alt_ptms["Mapping Result"] = np.nan
    ###success = gene location conserved and residue matches
    success = alt_ptms['Residue'] == alt_ptms['Canonical Residue']
    alt_ptms.loc[success, 'Mapping Result'] = 'Success'
    ###residue mismatch = gene location conserved and in frame, but residue does not match
    mismatch = (alt_ptms['Frame'].apply(lambda x: False if x != x else float(x) == 1)) & (alt_ptms['Residue'] != alt_ptms['PTM'].apply(lambda x: x.split('_')[1][0]))
    alt_ptms.loc[mismatch, 'Mapping Result'] = 'Residue Mismatch'
    ###frameshift = gene location conserved but ptm site no longer in frame
    frameshift = alt_ptms['Frame'].apply(lambda x: False if x != x else float(x) > 1)
    alt_ptms.loc[frameshift, 'Mapping Result'] = 'Different Reading Frame'
    ###ragged insertion = ptm exists at boundary and changes as a result of shifting boundary
    ragged_insertion = (alt_ptms['Ragged']) & (alt_ptms['Residue'] != alt_ptms['Canonical Residue'])
    alt_ptms.loc[ragged_insertion, 'Mapping Result'] = 'Ragged Insertion'
    ###noncoding region = gene location conserved, but is now in noncoding region (usually due to alternative promoter)
    alt_ptms.loc[alt_ptms['Frame'] == -1, 'Mapping Result'] = 'Noncoding Region'
    no_coding = ((~alt_ptms['Exon ID (Alternative)'].isna()) & (alt_ptms['Frame'].isna()))
    alt_ptms.loc[no_coding, 'Mapping Result'] = 'Found, But Missing Coding Info'
    ###not found = gene location is not conserved
    alt_ptms.loc[alt_ptms['Exon ID (Alternative)'].isna(), 'Mapping Result'] = 'Not Found'
    
    
    return alt_ptms
    
def annotateAlternativePTMs(all_ptms):
    #grab ptms mapped to alternative exons/transcripts
    alternative_ptms = [all_ptms['Type'] == 'Alternative']
    
    #rename columns into more meaningful labels
    alternative_ptms = alternative_ptms.rename({'Exon stable ID': 'Exon ID (Alternative)', 'Source Exon': 'Exon ID (Canonical)', 'Transcript stable ID': 'Alternative Transcript'}, axis = 1)
    alternative_ptms = alternative_ptms.drop('Type', axis = 1)
    
    #identify cases where ptms were successfully or unsuccessfully mapped
    alternative_ptms["Mapping Result"] = np.nan
    ###success = gene location conserved and residue matches
    success = alternative_ptms['Residue'] == alternative_ptms['PTM'].apply(lambda x: x.split('_')[1][0])
    alternative_ptms.loc[success, 'Mapping Result'] = 'Success'
    ###residue mismatch = gene location conserved and in frame, but residue does not match
    mismatch = (alternative_ptms['In Frame'].apply(lambda x: False if x != x else float(x) == 1)) & (alternative_ptms['Residue'] != alternative_ptms['PTM'].apply(lambda x: x.split('_')[1][0]))
    alternative_ptms.loc[mismatch, 'Mapping Result'] = 'Residue Mismatch'
    ###frameshift = gene location conserved but ptm site no longer in frame
    frameshift = alternative_ptms['In Frame'].apply(lambda x: False if x != x else float(x) > 1)
    alternative_ptms.loc[frameshift, 'Mapping Result'] = 'Different Reading Frame'
    ###ragged insertion = ptm exists at boundary and changes as a result of shifting boundary
    ragged_insertion = (alternative_ptms['Ragged']) & (alternative_ptms['Residue'] != alternative_ptms['PTM'].apply(lambda x: x.split('_')[1][0]))
    alternative_ptms.loc[ragged_insertion, 'Mapping Result'] = 'Ragged Insertion'
    ###noncoding region = gene location conserved, but is now in noncoding region (usually due to alternative promoter)
    alternative_ptms.loc[alternative_ptms['In Frame'] == -1, 'Mapping Result'] = 'Noncoding Region'
    no_coding = ((~alternative_ptms['Exon stable ID'].isna()) & (alternative_ptms['In Frame'].isna()))
    alternative_ptms.loc[no_coding, 'Mapping Result'] = 'Found, But Missing Coding Info'
    ###not found = gene location is not conserved
    alternative_ptms.loc[alternative_ptms['Exon stable ID'].isna(), 'Mapping Result'] = 'Not Found'
    
    return alternative_ptms
    
def getDistanceToBoundary(ptm_exon, gene_loc, strand):
    if strand == 1:
        distance_to_bound = ptm_exon['Exon End (Gene)'] - gene_loc
    else:
        distance_to_bound = gene_loc - ptm_exon['Exon Start (Gene)']
    return distance_to_bound
    
def getGenomicCoordinates(chromosome, gene_loc, strand):
    if strand == 1:
        return f"chr{chromosome}:{gene_loc}-{gene_loc+2}:{strand}"
    elif strand == -1:
        return f'chr{chromosome}:{gene_loc-2}-{gene_loc}:{strand}'
        
def getRaggedCoordinates(chromosome, gene_loc, ragged_loc, distance_to_bound, strand):
    if strand == 1:
        if distance_to_bound == 1:
            coords = f"chr{chromosome}:{gene_loc}-{gene_loc+1},{ragged_loc}:{strand}"
        else:
            coords = f"chr{chromosome}:{gene_loc},{ragged_loc}-{int(ragged_loc)+1}:{strand}"
    else:
        if distance_to_bound == 1:
            coords = f"chr{chromosome}:{int(ragged_loc)-1}-{ragged_loc},{gene_loc}:{strand}"
        else:
            coords = f"chr{chromosome}:{ragged_loc},{gene_loc-1}-{gene_loc}:{strand}"
    return coords
    
    
def getPTMLoc(ptm_exon, mapper, gene_loc, strand):
    #make sure transcript associated with exon contains coding information
    transcript = mapper.transcripts.loc[ptm_exon['Transcript stable ID']]
    if transcript['Relative CDS Start (bp)'] != 'No coding sequence' and transcript['Relative CDS Start (bp)'] != 'error:no match found':
        #check frame: if in frame, return residue and position
        frame, residue, aa_pos = utility.checkFrame(ptm_exon, transcript, gene_loc, 
                                 loc_type = 'Gene', strand = strand, return_residue = True)
        return frame, residue, aa_pos
    else:
        return np.nan, np.nan, np.nan