import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from cogent3 import make_aligned_seqs
import os
import gzip
import re
import sys
import time
import multiprocessing
from tqdm import tqdm
import pyliftover
from ExonPTMapper import config, processing, utility, alternative_mapping, get_splice_events


class PTM_mapper:
    def __init__(self):
        """
        Class for identifying relevant information about PTMs, and maps them to corresponding exons, transcripts, and genes
        
        Parameters
        ----------
        exons: pandas dataframe
            contains exon specific information for all transcripts in the genome (obtained from processing.py)
        transcripts: pandas dataframe
            contains transcript specific information for all transcripts in the genome (obtained from processing.py)
        ptms: pandas dataframe
            If None, will fully regenerate the minimal ptm specific information and initialize the class. If dataframe, will use information
               from dataframe to initialize class variables instead (dataframe should be generated previously using mapper class)
               
        Returns
        -------
        mapper class
        """
        self.load_PTMmapper()

        
    
    def find_ptms(self, transcript_id):
        """
        Given a transcript id, find all PTMs present in the protein
        
        Parameters
        ----------
        transcript_id: strings
            Ensemble transcript for the protein of interest
            
        Returns
        -------
        ptm_df: pandas dataframe
            Dataframe containing gene id, transcript id, protein id, residue modified, location of residue and modification type. Each row
                corresponds to a unique ptm
        """
        uniprot_id = config.translator.loc[config.translator['Transcript stable ID']==transcript_id, 'UniProtKB/Swiss-Prot ID'].values[0]
        gene_id = self.transcripts.loc[transcript_id, 'Gene stable ID']
        ptms = config.ps_api.get_PTMs(uniprot_id)
        
        #extract ptm position
        if isinstance(ptms, int):
            ptm_df = None
        else: 
            ptm_df = pd.DataFrame(ptms)
            ptm_df.columns = ['PTM Location (AA)', 'Residue', 'Modification']
            ptm_df.insert(0, 'Protein', uniprot_id)
            ptm_df.insert(0, 'Transcript', transcript_id)
            ptm_df.insert(0, 'Gene', gene_id)
            #ptm_df.index = ptm_df['Protein']+'_'+ptm_df['Residue']+ptm_df['PTM Location (AA)']
       
        return ptm_df
        
    def findAllPTMs(self, collapse = True, PROCESSES = 1):
        """
        Run find_ptms() for all transcripts, save in ptm_info dataframe
        """
        if PROCESSES == 1:
            df_list = []
            for trans in tqdm(config.available_transcripts, desc = 'Finding PTMs for available transcripts'):
                #check to make sure transcript has appropriate information
                if self.transcripts.loc[trans, 'Relative CDS Start (bp)'] != 'error:no match found' and self.exons[self.exons['Transcript stable ID'] == trans].shape[0] != 0:
                    info = self.find_ptms(trans)
                    if isinstance(info, str):		   
                        df_list.append(info)
                    elif info is None:
                        print(f'No PTMs found for {trans}')
                        continue
                    else:
                        df_list.append(info)		
                   
            self.ptm_info = pd.concat(df_list).dropna(axis = 1, how = 'all')
            self.ptm_info['PTM'] = self.ptm_info['Protein']+'_'+self.ptm_info['Residue']+self.ptm_info['PTM Location (AA)']
            if collapse:
                #collapse rows with duplicate indexes, but different transcripts or modififications into a single row, with each transcript or modification seperated by comma
                grouped_info = self.ptm_info.groupby(['Protein', 'Residue', 'PTM Location', 'PTM'])[['Gene', 'Transcript']].agg(','.join).reset_index()
                grouped_mods = self.ptm_info.groupby(['Protein', 'Residue', 'PTM Location', 'PTM']).agg(lambda x: ';'.join(x.unique())).reset_index()
                self.ptm_info = pd.merge([grouped_mods, grouped_info], on = ['Protein', 'Residue', 'PTM Location', 'PTM'])
                self.ptm_info = self.ptm_info.drop_duplicates()
                self.ptm_info = self.ptm_info.reset_index()
                self.ptm_info.index = self.ptm_info['PTM']
            else:
                self.ptm_info['PTM Location (AA)'] = self.ptm_info['PTM Location (AA)'].astype(int)
                self.ptm_info.index = self.ptm_info['PTM']
                self.ptm_info = self.ptm_info.drop_duplicates()
        else:
            print('Multiprocessing not active yet. Please use PROCESSES = 1')
            
    def mapPTM_singleTranscript(self, ptm_position, tid):
        """
        Given the location of a PTM in a protein and the transcript associated with the canonical protein, map the PTM to its exon and location in the genome.
        """
        if self.transcripts.loc[tid, 'Relative CDS Start (bp)'] == 'error:no match found':
            return np.repeat('CDS fail', 10)
        elif self.exons[self.exons['Transcript stable ID'] == tid].shape[0] == 0:
            return np.repeat('Exons Not Found', 10)
        else:
            CDS_start = int(self.transcripts.loc[tid, 'Relative CDS Start (bp)'])
    
            #calculate location of PTM in transcript (aa pos * 3 -3 + start of coding sequence)
            PTM_start = CDS_start + (ptm_position*3-3)
            

            #find which exon
            exon_info = self.exons.loc[self.exons['Transcript stable ID'] == tid]
            exon_row = (PTM_start < exon_info['Exon End (Transcript)']) & (PTM_start >= exon_info['Exon Start (Transcript)'])
            exon_of_interest = exon_info[exon_row].squeeze()
            exon_id = exon_of_interest['Exon stable ID']
            exon_rank = str(exon_of_interest['Exon rank in transcript'])
            
            #calculate distance to boundary, determine if it is a ragged site
            nterm_distance = PTM_start - exon_of_interest['Exon Start (Transcript)']
            cterm_distance = exon_of_interest['Exon End (Transcript)'] - (PTM_start + 3)
            min_distance = min([int(nterm_distance), int(cterm_distance)])
            ragged = min_distance < 0
    
            #find location in exon and gene
            exon_codon_start = PTM_start - int(exon_of_interest['Exon Start (Transcript)'])
            strand = self.genes.loc[self.transcripts.loc[tid, 'Gene stable ID'], 'Strand']
            if strand == 1:
                gene_codon_start = str(exon_codon_start + int(exon_of_interest['Exon Start (Gene)']))
            else:
                gene_codon_start = str(int(exon_of_interest['Exon End (Gene)']) - exon_codon_start)

                
            #find aa position in exon
            if exon_of_interest['Exon Start (Protein)'] == 'Partial start codon':
                exon_aa_start = 'Translation error'
            else:
                exon_aa_start = str(ptm_position - float(exon_of_interest['Exon Start (Protein)']))
                
            return gene_codon_start, str(PTM_start), str(exon_codon_start), exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance, min_distance, ragged
        
        
    def mapPTM(self, ptm):
        """
        Given a ptm (in the form of 'UniprotID_ResiduePosition'), find where the start of the codon producing the residue is found in the exon, transcript, and gene.
        
        Parameters
        ----------
        ptm: strings
            ptm to map to genome. Example: 'P00533_Y1042'
        
        Returns
        -------
        PTM_start: list or string 
            location in transcript of codon associated with PTM residue
        exon_id: list or string 
            exon ensemble id for the exon that the PTM codon is found
        exon_codon_start: list or string
            location in exon of codon associated with PTM residue
        gene_codon_start: list or string (depending on the number of transcripts/genes the protein is associated with
            location in gene of codon associated with PTM residue
        exon_aa_start: list or string
            residue number of the PTM within the exon. In other words, the number of amino acids from the start of the exon. Can be fractional.
        nterm_distance: list or string
            distance from the n-terminal/5' splice boundary, in base pairs
        cterm_distance: list or string
            distance from the c-terminal/3' splice boundary, in base pairs
        
        """
        #get necesary info
        position = self.ptm_info.loc[ptm, 'PTM Location (AA)']
        transcript_ids = self.ptm_info.loc[ptm, 'Transcripts'].split(',')

        if len(transcript_ids) > 1:
            PTM_start = []
            exon_id = []
            exon_rank = []
            exon_codon_start = []
            gene_codon_start = []
            nterm_distance = []
            cterm_distance = []
            exon_aa_start = []
            min_distance = []
            ragged = []
            for t in transcript_ids:
                map_results = self.mapPTM_singleTranscript(position, t)
                gene_codon_start.append(map_results[0])
                PTM_start.append(map_results[1])
                exon_codon_start.append(map_results[2])
                exon_aa_start.append(map_results[3])
                exon_id.append(map_results[4])
                exon_rank.append(map_results[5])
                nterm_distance.append(str(map_results[6]))
                cterm_distance.append(str(map_results[7]))
                min_distance.append(str(map_results[8]))
                ragged.append(str(map_results[9]))
                
            #convert lists to strings
            PTM_start = ','.join(PTM_start)   
            exon_id = ','.join(exon_id) 
            exon_rank = ','.join(exon_rank)
            nterm_distance = ','.join(nterm_distance)
            cterm_distance = ','.join(cterm_distance)
            min_distance = ','.join(min_distance)
            exon_codon_start = ','.join(exon_codon_start)
            gene_codon_start = ','.join(gene_codon_start)
            exon_aa_start = ','.join(exon_aa_start)
            ragged = ','.join(ragged)
        else:
            map_results = self.mapPTM_singleTranscript(position, transcript_ids[0])
            gene_codon_start, PTM_start, exon_codon_start, exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance, min_distance, ragged = map_results

        
        return pd.Series(data = [gene_codon_start, PTM_start, exon_codon_start, exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance, min_distance, ragged],
                        index = ['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon rank in transcript', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged'],
                        name = ptm)
                        
        
    def mapPTMs_all(self, save_iter = 5000, restart = False, PROCESSES = 1):
        """
        For all ptms in ptm_info, map to genome and save in ptm_positions 
        """
        #create copy of existing ptm_info dataframe
        ptm_info = self.ptm_info.copy()
        print('Getting location of PTMs in transcript')
        #extract transcript level info required for mapping process (coding start and location of splice boundaries)
        transcript_data = self.transcripts[['Relative CDS Start (bp)', 'Exon cuts']].copy()
        transcript_data = transcript_data.dropna(subset = ['Exon cuts', 'Relative CDS Start (bp)'])
        #convert exon cuts into list containing integers rather than a single string
        transcript_data['Exon cuts'] = transcript_data['Exon cuts'].apply(lambda cut: np.array([int(x) for x in cut.split(',')]))
        transcript_data['Relative CDS Start (bp)'] = pd.to_numeric(transcript_data['Relative CDS Start (bp)'], errors = 'coerce')
        transcript_data = transcript_data.dropna(subset = ['Exon cuts', 'Relative CDS Start (bp)'])
        #add transcript data to ptm information
        ptm_info = ptm_info.merge(transcript_data, left_on = 'Transcript', right_index = True, how = 'left')
        ptm_info = ptm_info.dropna(subset = 'Exon cuts')

        #get transcript location of PTMs
        ptm_info['Transcript Location (NC)'] = ((ptm_info['PTM Location (AA)']-1)*3 + ptm_info['Relative CDS Start (bp)']).astype(int)
        
        
        #get rank of exon in transcript, based on transcript location and exon cuts. To do so, find the first exon cut which is greater than transcript location.
        min_exon_rank = self.exons.groupby('Transcript stable ID')['Exon rank in transcript'].min()
        exon_rank = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Identify PTM-containing exons'):
            #get minimum exon rank in transcript, for rare case when first exons don't have sequence info
            #min_exon_rank = self.exons.loc[self.exons['Transcript stable ID'] == row['Transcript'], 'Exon rank in transcript'].min()
            normed_cuts = row['Transcript Location (NC)'] - row['Exon cuts']
            #find the first negative number in normed_cuts (larger than transcript loc)
            for c in range(len(normed_cuts)):
                if normed_cuts[c] <  0:
                    #add 1, as currently in pythonic coordinates (if missing first exon ranks, add additional)
                    exon_rank.append(c+min_exon_rank[row['Transcript']])
                    break
                    
        ptm_info['Exon rank in transcript'] = exon_rank
        ptm_info = ptm_info.drop('Exon cuts', axis = 1)
        
        #add exon level information required for rest of mapping process
        exon_info = self.exons[['Transcript stable ID', 'Exon stable ID', 'Exon rank in transcript', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)', 'Exon Start (Protein)']].copy()
        ptm_info = ptm_info.merge(exon_info, left_on = ['Transcript', 'Exon rank in transcript'], right_on = ['Transcript stable ID', 'Exon rank in transcript'], how = 'left')
        print('Getting distance to exon-exon junctions')
        #get distance to exon-exon junctions and start of PTM in exon
        ptm_info['Distance to N-terminal Splice Boundary (NC)'] = (ptm_info['Transcript Location (NC)'] - ptm_info['Exon Start (Transcript)'])
        ptm_info['Distance to C-terminal Splice Boundary (NC)'] = (ptm_info['Exon End (Transcript)'] - (ptm_info['Transcript Location (NC)']+3))
        min_dist = []
        for i, row in ptm_info.iterrows():
            min_dist.append(min([row['Distance to N-terminal Splice Boundary (NC)'], row['Distance to C-terminal Splice Boundary (NC)']]))
        ptm_info['Distance to Closest Boundary (NC)'] = min_dist
        ptm_info['Ragged'] = ptm_info['Distance to Closest Boundary (NC)'] < 0
        
        #get genomic location of ptms and coordinates (check which strand gene is on, will change how this is calculated)
        ptm_info = ptm_info.merge(self.genes[['Chromosome/scaffold name', 'Strand']], left_on = 'Gene', right_index = True)
        gene_loc = []
        coordinates = []
        second_exon = []
        ragged_loc_list = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Getting location of PTMs in genome'):
            if row['Strand'] == 1:
                loc = row['Distance to N-terminal Splice Boundary (NC)'] + row['Exon Start (Gene)']
            else:
                loc = row['Exon End (Gene)'] - row['Distance to N-terminal Splice Boundary (NC)']
            #check if able to get location, if so convert to integer
            if loc == loc:
                loc = int(loc)
            gene_loc.append(loc)
                
            if not row['Ragged']:
                coordinates.append(getGenomicCoordinates(row['Chromosome/scaffold name'], loc, row['Strand']))
                second_exon.append(np.nan)
                ragged_loc_list.append(np.nan)
            else:
                #identify the other exon contributing to ragged PTM site, get start of this exon 
                next_rank = row['Exon rank in transcript'] + 1
                transcript = self.exons[self.exons['Transcript stable ID'] == row['Transcript']]
                next_exon = transcript[transcript['Exon rank in transcript'] == next_rank].squeeze()
                ragged_loc = int(next_exon['Exon Start (Gene)'] if row['Strand'] == 1 else next_exon['Exon End (Gene)'])
                #get coordinates of ragged PTM, save exon id of second contributing exon
                coordinates.append(getRaggedCoordinates(row['Chromosome/scaffold name'], loc, ragged_loc, min_dist, row['Strand']))
                ragged_loc_list.append(ragged_loc)
                second_exon.append(next_exon['Exon stable ID'])
        ptm_info['Gene Location (NC)'] = gene_loc
        ptm_info['Genomic Coordinates'] = coordinates
        ptm_info['Second Contributing Exon'] = second_exon
        ptm_info['Ragged Genomic Location'] = ragged_loc_list
        ptm_info = ptm_info.drop(['Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)'], axis = 1)
        

        
        #get ptm location in exon in amino acid coordinates
        exon_aa_loc = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Getting residue number within each exon'):
            if row['Exon Start (Protein)'] == 'Partial start codon' or row['Exon Start (Protein)'] == "5' NCR" or row['Exon Start (Protein)'] == "3' NCR":
                exon_aa_loc.append(np.nan)
            else:
                exon_aa_loc.append(row['PTM Location (AA)'] - float(row['Exon Start (Protein)']))
        ptm_info['Exon Location (AA)'] = exon_aa_loc
        
        print('Processing final mapped data')
        #save new dataframe which will be trimmed version of ptm info with each row containing a PTM mapped to unique genomic coordinates
        ptm_coordinates = ptm_info[['Genomic Coordinates', 'PTM','Residue', 'Modification', 'Chromosome/scaffold name', 'Strand','Gene Location (NC)', 'Ragged', 'Ragged Genomic Location', 'Exon stable ID']].copy()
        ptm_coordinates = ptm_coordinates.dropna(subset = 'Gene Location (NC)')
        ptm_coordinates = ptm_coordinates.drop_duplicates()
        ptm_coordinates = ptm_coordinates.astype({'Gene Location (NC)': int, 'Strand':int, 'Ragged':bool})
        #group modifications for the same ptm in the same row
        grouped = ptm_coordinates.groupby(['Genomic Coordinates', 'PTM', 'Chromosome/scaffold name', 'Residue', 'Strand', 'Gene Location (NC)', 'Ragged'])
        ptm_coordinates = pd.concat([grouped['Ragged Genomic Location'].apply(lambda x: np.unique(x)[0]), grouped['Modification'].agg(lambda x: ';'.join(np.unique(x))), grouped['Exon stable ID'].agg(lambda x: ';'.join(np.unique(x)))], axis = 1)
        ptm_coordinates = ptm_coordinates.reset_index()
        ptm_coordinates = ptm_coordinates.rename({'PTM':'Source of PTM', 'Modification':'Modifications', 'Exon stable ID': 'Source Exons'}, axis = 1)
        ptm_coordinates.index = ptm_coordinates['Genomic Coordinates']
        ptm_coordinates = ptm_coordinates.drop('Genomic Coordinates', axis = 1)
        #get coordinates in the hg19 version of ensembl using hg38 information
        hg19_coords = []
        liftover_object = pyliftover.LiftOver('hg38','hg19')
        for i, row in tqdm(ptm_coordinates.iterrows(), total = ptm_coordinates.shape[0], desc = 'Converting from hg38 to hg19 coordinates'):
            hg19_coords.append(convertToHG19(row['Gene Location (NC)'], row['Chromosome/scaffold name'], row['Strand'], liftover_object = liftover_object))
        ptm_coordinates['HG19 Location'] = hg19_coords
        ptm_coordinates = ptm_coordinates.drop_duplicates()
        self.ptm_coordinates = ptm_coordinates.copy()
        
        
        #collapse information for each ptm, save new ptm_info dataframe 
        ptm_info = ptm_info.drop(['Strand', 'Transcript stable ID', 'Relative CDS Start (bp)', 'Chromosome/scaffold name', 
                                'Second Contributing Exon', 'Ragged Genomic Location'], axis = 1)
        ptm_info = ptm_info.astype(str)
        ptm_info = ptm_info.groupby(['PTM','Protein', 'Residue', 'PTM Location (AA)']).agg(';'.join).reset_index()
        ptm_info['PTM Location (AA)'] = ptm_info['PTM Location (AA)'].astype(int)
        ptm_info = ptm_info.rename({'Transcript':'Transcripts', 'Gene':'Genes', 'Modification':'Modifications'}, axis = 1)
        ptm_info.index = ptm_info['PTM']
        ptm_info = ptm_info.drop('PTM', axis = 1)
        self.ptm_info = ptm_info.copy()
            

            
    def explode_PTMinfo(self, explode_cols = ['Genes', 'Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon rank in transcript']):
        exploded_ptms = self.ptm_info.copy()
        #check columns that exist
        explode_cols = [col for col in explode_cols if col in exploded_ptms.columns.values]
        #split different entries
        for col in explode_cols:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(';') if x == x else np.nan)
      
        exploded_ptms = exploded_ptms.explode(explode_cols)
        exploded_ptms = exploded_ptms.reset_index()
        exploded_ptms = exploded_ptms.rename({'index':'PTM'}, axis = 1)
        return exploded_ptms
        
    def collapse_PTMinfo(self, all_cols = ['Genes', 'Transcripts','Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)','Exon stable ID', 'Exon rank in transcript'], unique_cols = ['Modifications', 'Exon Location (AA)', 'Protein', 'PTM Location (AA)', 'Residue']):
        ptms = self.ptm_info.copy()
        ptms = ptms.astype(str)
        grouped_mods = ptms.groupby(level = 0)
        collapsed_ptms = []
        for col in self.ptm_info.columns:
            if col in unique_cols:
                collapsed_ptms.append(grouped_mods[col].apply(set).apply(';'.join))
            else:
                collapsed_ptms.append(grouped_mods[col].apply(';'.join))
        collapsed_ptms = pd.concat(collapsed_ptms, axis = 1)
        return collapsed_ptms

        
    def getTrypticFragment(self, pos, transcript_id):
        """
        Given a ptm, find the tryptic fragment you would be likely to find from mass spec (trypsin cuts at lysine or arginine (except after proline). 
        
        Parameters
        ----------
        ptm: str
            ptm to get tryptic fragment for, indicated by 'SubstrateID_SiteNum'
        
        Returns
        -------
        seq: str
            tryptic fragment sequence
        """
        seq = self.transcripts.loc[transcript_id, 'Amino Acid Sequence']
        c_terminal_cut = re.search('[K|R][^P]|$K|R', seq[pos:])
        if c_terminal_cut is None:
            c_terminal_cut = len(seq)
        else:
            c_terminal_cut = pos + 1+ c_terminal_cut.span()[0]
            
        n_terminal_cut = re.search('^K|R|[^P][K|R]', seq[pos-2::-1])
        if n_terminal_cut is None:
            n_terminal_cut = 0
        else:
            n_terminal_cut = pos - n_terminal_cut.span()[1]
            
        
        return seq[n_terminal_cut:c_terminal_cut]
    
    def getAllTrypticFragments(self):
        """
        Runs getTrypticFragment() for all ptms recorded in self.ptm_info. Adds 'Tryptic Fragment' column to self.ptm_info after running.
        """
        fragment = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Getting tryptic fragments'):
            pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
            transcript = self.ptm_info.loc[ptm, 'Transcripts']
            #if multiple transcripts associated with protein, only use first transcript (should be same seq)
            if ';' in transcript:
                transcript = transcript.split(';')[0]
            fragment.append(self.getTrypticFragment(pos, transcript))
        self.ptm_info['Tryptic Fragment'] = fragment
        
    def getFlankingSeq(self, pos, transcript, flank_size = 10):
        """
        Get flanking sequence around the indicated PTM, with the number of residues upstream/downstream indicated by flank_size
        
        Parameters
        ----------
        ptm: str
            ptm to get flanking sequence, indicated by 'SubstrateID_SiteNum'
        flank_size: int
            number of residues to return around the ptm
            
        Returns
        -------
        flank_seq: str
            flanking sequence around the ptm of interest
        """
        #if multiple transcripts associated with protein, only use first transcript (should be same seq)
        pos = int(pos)
        protein_sequence = self.transcripts.loc[transcript, 'Amino Acid Sequence']
        if pos <= flank_size:
            #if amino acid does not have long enough N-terminal flanking sequence, add spaces to cushion sequence
            spaces = ''.join([' ' for i in range(flank_size - pos + 1)])
            flank_seq = spaces + protein_sequence[0:pos-1]+protein_sequence[pos-1].lower()+protein_sequence[pos:pos+flank_size]
        elif len(protein_sequence)-pos <= flank_size:
            #if amino acid does not have long enough C-terminal flanking sequence, add spaces to cushion sequence
            spaces = ''.join([' ' for i in range(flank_size - (len(protein_sequence)-pos))])
            flank_seq = protein_sequence[pos-flank_size-1:pos-1]+protein_sequence[pos-1].lower()+protein_sequence[pos:]+spaces
        else:
            #full flanking sequence available
            flank_seq = protein_sequence[pos-flank_size-1:pos-1]+protein_sequence[pos-1].lower()+protein_sequence[pos:pos+flank_size]
        return flank_seq
        
    def getAllFlankingSeqs(self, flank_size = 10):
        """
        Runs getAllFlankingSeqs() for all ptms recorded in self.ptm_info. Adds 'Flanking Sequence' column to self.ptm_info after running.
        """
        flanks = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Get flanking sequences'):
            pos = self.ptm_info.loc[ptm, 'PTM Location (AA)']
            transcript = self.ptm_info.loc[ptm, 'Transcripts']
            #if multiple transcripts associated with protein, only use first transcript (should be same seq)
            if ';' in transcript:
                transcript = transcript.split(';')[0]
            flanks.append(self.getFlankingSeq(pos, transcript, flank_size))
        self.ptm_info['Flanking Sequence'] = flanks
            
        
    def findFlankingSeq(self, transcript_id, flank_seq):
        """
        Given a transcript ID, find where the provided flanking sequence is located in the transcript. Can be used to see whether a ptm (and its flanking residues) are conserved within the transcript.
        
        Parameters
        ----------
        transcript_id: str
            Ensemble transcript ID for transcript of interest
        flank_seq: str
            flanking sequence surrounding the ptm of interest
        
        Returns
        -------
        indicates the location of the ptm in the coded transcript (by residue)
        """
        full_seq = str(self.transcripts.loc[self.transcripts['Transcript stable ID'] == transcript_id, 'Amino Acid Sequence'].values[0])
        flank_len = (len(flank_seq)-1)/2
        match = re.search(str(flank_seq), full_seq)
        if match:
            #check to see if any other matches exist
            if re.search(str(flank_seq), full_seq[match.span()[1]:]):
                return str(match.span()[0]+1+flank_len) + '(multiple matches)'
            else:
                return '('+str(match.span()[0]+1+flank_len)+')'
        else:
            return np.nan
            
    def findInDomains(self, ptm):
        """
        Given a ptm, figure out whether the ptm is located in a domain.
        
        Parameters
        ----------
        ptm: str
            ptm to get flanking sequence, indicated by 'SubstrateID_SiteNum'
        
        Returns
        -------
        inDomain: bool
            indicates whether ptm is located in a domain
        domain_type: str
            if in domain, indicates the type of domain. If not in domain, returns np.nan.
        """
        protein = self.ptm_info.loc[ptm, 'Protein']
        pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
        domains = config.ps_api.get_domains(protein, 'uniprot')
        inDomain = False
        domain_type = np.nan
        for dom in domains:
            start = int(dom[1])
            stop = int(dom[2])
            if pos >= start and pos <= stop:
                inDomain = True
                domain_type = dom[0]
                break
        return inDomain, domain_type
    
    def findAllinDomains(self):
        """
        Run findInDomain() for all ptms in self.ptm_info and save the results in self.ptm_info under 'inDomain' and 'Domain Type' columns.
        """
        inDomain_list = []
        domain_type_list = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Finding location in a domain'):
            results = self.findInDomains(ptm)
            inDomain_list.append(results[0])
            domain_type_list.append(results[1])
            
        self.ptm_info['inDomain'] = inDomain_list
        self.ptm_info['Domain Type'] = domain_type_list
        
    def mapPTMtoExons(self, ptm, trim_exons = None, alternative_only = True):
            
        #reduce exons dataframe to exons associated with transcripts with available information, if not provided
        if trim_exons is None:
            #identify transcripts (plus associated exons) with available transcript and amino acid sequence
            available_transcripts = self.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
            #if desired, isolate only alternative transcripts
            if alternative_only:
                alternative_transcripts = config.translator.loc[config.translator['Uniprot Canonical'] != 'Canonical', 'Transcript stable ID']
                available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
                
            trim_exons = self.exons[self.exons['Transcript stable ID'].isin(available_transcripts)]
            #extract only exon information required for mapping, making sure no duplicates
            trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()
            #add chromosome and strand info to list
            trim_exons = trim_exons.merge(self.genes[['Chromosome/scaffold name', 'Strand']], left_on = 'Gene stable ID', right_index = True)
        
        results = None
        #isolate info into unique genes (if multiple associated with protein/ptm of interest)
        coordinates = ptm.name
        gene_loc = ptm['Gene Location (NC)']
        ragged = ptm['Ragged']
        ragged_loc = ptm['Ragged Genomic Location']
        chromosome = ptm['Chromosome/scaffold name']
        strand = ptm['Strand']
        #restrict exons to those on the same chromosome and strand
        trim_exons = trim_exons[(trim_exons['Chromosome/scaffold name'] == chromosome) &(trim_exons['Strand'] == strand)]
        
        #grab all exons containing ptm site (based on genome location and ensuring they come from same gene info)
        ptm_exons = trim_exons[(trim_exons['Exon Start (Gene)'] <= gene_loc) & (trim_exons['Exon End (Gene)'] >= gene_loc)].copy()
        
        #if site is ragged, check second exon for conserved region
        if utility.stringToBoolean(ragged):
            #check if second exon is conserved AND was not already found when looking for first exon
            second_ptm_exons = trim_exons[(trim_exons['Exon Start (Gene)'] <= ragged_loc) & (trim_exons['Exon End (Gene)'] >= ragged_loc) & ~(trim_exons['Transcript stable ID'].isin(ptm_exons['Transcript stable ID']))].copy()
            if second_ptm_exons.shape[0] == 0:
                second_ptm_exons = None
        else:
            second_ptm_exons = None
            
        if ptm_exons.shape[0] != 0:
            ptm_exons['Source Exons'] = ptm['Source Exons']
            ptm_exons['Source of PTM'] = ptm['Source of PTM']
            ptm_exons['Canonical Residue'] = ptm['Residue']
            ptm_exons['Modifications'] = ptm['Modifications']
            
            #ptm_exons['Source Exon'] = source_exon_id
            

            #check if ptm is at the boundary (ragged site)
            distance_to_bound = ptm_exons.apply(getDistanceToBoundary, args = (gene_loc, strand), axis = 1)
            ptm_exons['Ragged'] = distance_to_bound <= 1
            #identify location of ptm in new exons (different equations depending on if gene is on the forward or reverse strand)
            if not ptm_exons['Ragged'].any():
                ptm_exons['Genomic Coordinates'] = coordinates
                ptm_exons['Second Exon'] = np.nan
                #check if frame matches for given event: if it does, identify new residue + position in protein associated with exon
                ptm_loc_info = ptm_exons.apply(getPTMLoc, args = (self, gene_loc, strand), axis = 1)
                #unpack results and save to dataframe
                ptm_exons['Frame'] = [x[0] for x in ptm_loc_info]
                ptm_exons['Alternative Residue'] = [x[1] for x in ptm_loc_info]
                ptm_exons['Alternative Protein Position (AA)'] = [x[2] for x in ptm_loc_info]
            else:
                coords = []
                aa_pos_list = []
                residue_list = []
                frame_list = []
                second_exon_list = []
                for i, row in ptm_exons.iterrows():
                    if row['Ragged']:
                        next_rank = row['Exon rank in transcript'] + 1
                        transcript = self.exons[self.exons['Transcript stable ID'] == row['Transcript stable ID']]
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
                            frame, residue, aa_pos = getPTMLoc(row, self, gene_loc, strand)
                            frame_list.append(frame)
                            residue_list.append(residue)
                            aa_pos_list.append(aa_pos)
                            second_exon_list.append(next_exon['Exon stable ID'])
                            #get genomic coordinates (from both contributing exons)
                            coords.append(getRaggedCoordinates(chromosome, gene_loc, ragged_loc, distance_to_bound[i], strand))

                    else:
                        coords.append(getGenomicCoordinates(chromosome, gene_loc, strand))
                        #check if frame matches for given event: if it does, identify new residue + position in protein associated with exon
                        frame, residue, aa_pos = getPTMLoc(row, self, gene_loc,strand)
                        frame_list.append(frame)
                        residue_list.append(residue)
                        aa_pos_list.append(aa_pos)
                        second_exon_list.append(np.nan)

                ptm_exons['Genomic Coordinates'] = coords
                ptm_exons['Frame'] = frame_list
                ptm_exons['Alternative Residue'] = residue_list
                ptm_exons['Alternative Protein Position (AA)'] = aa_pos_list
                ptm_exons['Second Exon'] = second_exon_list

            ptm_exons['Ragged'] = ptm_exons['Ragged'].astype(str)
                
        if second_ptm_exons is not None:
            coordinates = []
            frame_list = []
            residue_list = []
            position_list = []
            for i, row in second_ptm_exons.iterrows():
                #get PTM distance to boundary (should be the same for all canonical transcripts). this will indicate how much each exon contributes to ragged site
                dist_to_boundary =  int(self.ptm_info.loc[ptm['Source of PTM'], 'Distance to Closest Boundary (NC)'].split(';')[0])
                start_second_exon = row['Exon Start (Transcript)']
                codon_start = start_second_exon - (3+dist_to_boundary)
                
                #check frame
                transcript = self.transcripts.loc[row['Transcript stable ID']]
                if transcript['Relative CDS Start (bp)'] != 'No coding sequence' and transcript['Relative CDS Start (bp)'] != 'error:no match found':
                    frame, residue, aa_pos = utility.checkFrame(row, transcript, codon_start, loc_type = 'Transcript', strand = strand, return_residue = True)
                else:
                    frame, residue, aa_pos = np.repeat(np.nan, 3)
                #transcript_sequence = transcripts.loc[transcripts[g], 'Transcript Sequence']
                #codon = transcript_sequence[codon_start:codon_start+3]
                #new_residue = utility.codon_dict[codon]
                #residue_list.append(new_residue)

                    
                frame_list.append(frame)
                residue_list.append(residue)
                position_list.append(aa_pos)
                ragged_loc = row['Exon Start (Gene)'] if strand == 1 else row['Exon End (Gene)']
                coordinates.append(getRaggedCoordinates(chromosome, gene_loc, ragged_loc, dist_to_boundary, strand))

                    
            second_ptm_exons['Genomic Coordinates'] = coordinates
            second_ptm_exons['Frame'] = frame_list
            second_ptm_exons['Alternative Residue'] = residue_list
            second_ptm_exons['Alternative Protein Position (AA)'] = position_list
            second_ptm_exons = second_ptm_exons.rename({'Exon stable ID': 'Second Exon'}, axis = 1)
            second_ptm_exons['Source Exons'] = ptm['Source Exons']
            second_ptm_exons['Source of PTM'] = ptm['Source of PTM']
            second_ptm_exons['Modifications'] = ptm['Modifications']
            second_ptm_exons['Ragged'] = True
            second_ptm_exons['Ragged'] = second_ptm_exons['Ragged'].astype(str)
            ptm_exons = pd.concat([ptm_exons, second_ptm_exons])

        if ptm_exons.shape[0] > 0:
            results = ptm_exons.copy()
                    
        return results
        
    def mapPTMsToAlternativeExons(self):
        #get all alternative transcripts with available coding info
        available_transcripts = self.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
        alternative_transcripts = config.translator.loc[config.translator['Uniprot Canonical'] != 'Canonical', 'Transcript stable ID']
        available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
        
        #grab exons associated with available transcripts
        trim_exons = self.exons[self.exons['Transcript stable ID'].isin(available_transcripts)]
        trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()
        #add chromosome and strand information to exon dataframe
        trim_exons = trim_exons.merge(self.genes[['Chromosome/scaffold name', 'Strand']], left_on = 'Gene stable ID', right_index = True)
        
        #convert gene location into string
        if os.path.exists(config.processed_data_dir + 'temp_alt_ptms.csv'):
            alt_ptms = pd.read_csv(config.processed_data_dir + 'temp_alt_ptms.csv', dtype = {'Ragged': str})
            alt_ptms = alt_ptms.dropna(subset = 'Source of PTM')
            analyzed_ptms = alt_ptms['Source of PTM'].unique()
            print(f'Found {len(analyzed_ptms)} already analyzed from previous runs')
            ptms_to_analyze = self.ptm_coordinates[~self.ptm_coordinates['Source of PTM'].isin(analyzed_ptms)]
            i = 1
            for index,ptm in tqdm(ptms_to_analyze.iterrows(), total = ptms_to_analyze.shape[0]):
                ptm_exons = self.mapPTMtoExons(ptm, trim_exons = trim_exons)
                if ptm_exons is not None:
                    alt_ptms = pd.concat([alt_ptms, ptm_exons])
                
                if i % 10000 == 0:
                    alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
                i = i + 1
        else:
            alt_ptms = None
            i = 1
            for index,ptm in tqdm(self.ptm_coordinates.iterrows(), total = self.ptm_coordinates.shape[0]):
                ptm_exons = self.mapPTMtoExons(ptm, trim_exons = trim_exons)
                if ptm_exons is not None:
                    if alt_ptms is None:
                        alt_ptms = ptm_exons.copy()
                    else:
                        alt_ptms = pd.concat([alt_ptms, ptm_exons])
                        
                if i % 10000 == 0:
                    alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
                i = i + 1
                
        alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
        print('Saving mapped ptms, final')
        
        #remove residual exon columns that are not needed
        alt_ptms = alt_ptms.drop(['Exon End (Gene)', 'Exon Start (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)', 'Exon rank in transcript'], axis = 1)
        #grab ptm specific info
        alt_ptms['Canonical Protein Position (AA)'] = alt_ptms['Source of PTM'].apply(lambda x: x.split('_')[1][1:] if x == x else np.nan)
              
        #add ptms that were unsuccessfully mapped to alternative transcripts
        alternative = self.proteins.dropna(subset = 'Alternative Transcripts (All)').copy()
        alternative['Alternative Transcripts (All)'] = alternative['Alternative Transcripts (All)'].apply(lambda x: x.split(','))
        alternative = alternative.explode('Alternative Transcripts (All)').reset_index()
        alternative = alternative[['UniProtKB/Swiss-Prot ID', 'Alternative Transcripts (All)']]
        alternative = alternative[alternative['Alternative Transcripts (All)'].isin(available_transcripts)]
        alternative = alternative.rename({'UniProtKB/Swiss-Prot ID':'Protein', 'Alternative Transcripts (All)':'Transcript stable ID'}, axis = 1)
        ptms = self.ptm_info.reset_index()[['PTM', 'Protein', 'PTM Location (AA)', 'Exon stable ID', 'Ragged', 'Modifications', 'Residue']].drop_duplicates()
        ptms = ptms.rename({'PTM':'Source of PTM','PTM Location (AA)':'Canonical Protein Location (AA)', 'Residue':'Canonical Residue'}, axis = 1)
        alternative = alternative.merge(ptms, on = 'Protein')
        
        #identify ptms not present in isoforms
        potential_ptm_isoform_labels = alternative['Transcript stable ID'] + '_' + alternative['Source of PTM']
        mapped_ptm_isoform_labels = alt_ptms['Transcript stable ID'] + '_' + alt_ptms['Source of PTM']
        missing = alternative[~potential_ptm_isoform_labels.isin(mapped_ptm_isoform_labels)]
        missing = missing.rename({'Transcripts':'Transcript stable ID','Exon stable ID': 'Source Exons', 'PTM':'Source of PTM', 'Residue':'Canonical Residue'}, axis = 1)
        #add genes
        missing = missing.merge(self.transcripts['Gene stable ID'].reset_index(), on = 'Transcript stable ID', how = 'left')
        #add gene info
        missing = missing.merge(self.genes[['Chromosome/scaffold name', 'Strand']].reset_index(), on = 'Gene stable ID', how = 'left')
        alt_ptms = pd.concat([alt_ptms, missing])
        
       
        
        #rename columns
        alt_ptms = alt_ptms.rename({'Exon stable ID': 'Exon ID (Alternative)', 'Source Exons': 'Exon ID (Canonical)', 'Transcript stable ID': 'Alternative Transcript'}, axis = 1)
        
        #identify cases where ptms were successfully or unsuccessfully mapped
        alt_ptms["Mapping Result"] = np.nan
        ###success = gene location conserved and residue matches
        success = alt_ptms['Alternative Residue'] == alt_ptms['Canonical Residue']
        alt_ptms.loc[success, 'Mapping Result'] = 'Success'
        ###residue mismatch = gene location conserved and in frame, but residue does not match
        mismatch = (alt_ptms['Frame'].apply(lambda x: False if x != x else float(x) == 1)) & (alt_ptms['Alternative Residue'] != alt_ptms['Canonical Residue'])
        alt_ptms.loc[mismatch, 'Mapping Result'] = 'Residue Mismatch'
        ###frameshift = gene location conserved but ptm site no longer in frame
        frameshift = alt_ptms['Frame'].apply(lambda x: False if x != x else float(x) > 1)
        alt_ptms.loc[frameshift, 'Mapping Result'] = 'Different Reading Frame'
        ###ragged insertion = ptm exists at boundary and changes as a result of shifting boundary
        ragged_insertion = (alt_ptms['Ragged'].apply(lambda x: utility.stringToBoolean(x))) & (alt_ptms['Alternative Residue'] != alt_ptms['Canonical Residue'])
        alt_ptms.loc[ragged_insertion, 'Mapping Result'] = 'Ragged Insertion'
        ###noncoding region = gene location conserved, but is now in noncoding region (usually due to alternative promoter)
        alt_ptms.loc[alt_ptms['Frame'] == -1, 'Mapping Result'] = 'Noncoding Region'
        no_coding = ((~alt_ptms['Exon ID (Alternative)'].isna()) & (alt_ptms['Frame'].isna()))
        alt_ptms.loc[no_coding, 'Mapping Result'] = 'Found, But Missing Coding Info'
        ###not found = gene location is not conserved
        alt_ptms.loc[alt_ptms['Alternative Protein Position (AA)'].isna(), 'Mapping Result'] = 'Not Found'
        ###Ragged Insertion = missing ptm at a exon-exon junction
        alt_ptms.loc[(alt_ptms['Mapping Result'] == 'Not Found') & (alt_ptms['Ragged'] == 'True'), 'Mapping Result'] = 'Ragged Insertion'
        
        self.alternative_ptms = alt_ptms
        os.remove(config.processed_data_dir + 'tmp_alt_ptms.csv')
    
    def calculate_PTMconservation(self, transcript_subset = None, isoform_subset = None, save_col = 'PTM Conservation Score', save_transcripts = True, return_score = False, unique_isoforms = True):
        if unique_isoforms:
            if self.isoform_ptms is None:
                self.getIsoformSpecificPTMs()
            
            if isoform_subset is not None:
                alt_ptms = self.isoform_ptms[self.isoform_ptms['Isoform ID'].isin(isoform_subset)].copy()
            elif transcript_subset is not None:
                raise ValueError('When looking at unique isoforms, indicate the subset of isoforms uing isoform ids rather than transcript ids')
            else:
                alt_ptms = self.isoform_ptms.copy()
                
            conserved_transcripts = alt_ptms.dropna(subset = 'Alternative Residue').groupby('Source of PTM')['Isoform ID'].apply(list)
            lost_transcripts = alt_ptms[alt_ptms['Alternative Residue'].isna()].groupby('Source of PTM')['Isoform ID'].apply(list)
                
        else:
            if self.alternative_ptms is None:
                raise AttributeError('No alternative_ptms attribute. Must first map ptms to alternative transcripts with mapPTMsToAlternative()')
            else:
                if transcript_subset is not None:
                    alt_ptms = self.alternative_ptms[self.alternative_ptms['Alternative Transcript'].isin(transcript_subset)].copy()
                else:
                    alt_ptms = self.alternative_ptms.copy()
                    
            conserved_transcripts = alt_ptms[alt_ptms['Mapping Result'] == 'Success'].groupby('Source of PTM')['Alternative Transcript'].apply(list)
            lost_transcripts = alt_ptms[alt_ptms['Mapping Result'] != 'Success'].groupby('Source of PTM')['Alternative Transcript'].apply(list)
        
        #conserved transcripts
        num_conserved_transcripts = conserved_transcripts.apply(len)
        conserved_transcripts = conserved_transcripts.apply(','.join)
        
        #lost transcripts
        num_lost_transcripts = lost_transcripts.apply(len)
        lost_transcripts = lost_transcripts.apply(','.join)
        
        #save results in ptm_info
        if save_transcripts:
            self.ptm_info['Number of Conserved Transcripts'] = num_conserved_transcripts
            self.ptm_info['Conserved Transcripts'] = conserved_transcripts
            self.ptm_info['Number of Lost Transcripts'] = num_lost_transcripts
            self.ptm_info['Lost Transcripts'] = lost_transcripts
        
        #calculate fraction of transcripts which have conserved PTM
        conservation_score = []
        for ptm in self.ptm_info.index:
            num_conserved = self.ptm_info.loc[ptm,'Number of Conserved Transcripts']
            num_lost = self.ptm_info.loc[ptm,'Number of Lost Transcripts']
            #check if there are any conserved transcripts (or if not and is NaN)
            if num_conserved != num_conserved and num_lost == num_lost:
                conservation_score.append(0)
            elif num_conserved != num_conserved and num_lost != num_lost:
                conservation_score.append(1)
            #check if any lost transcripts: if not replace NaN with 0 when calculating
            elif num_lost != num_lost and num_conserved == num_conserved:
                conservation_score.append(1)
            else:
                conservation_score.append(num_conserved/(num_conserved+num_lost))
        
        self.ptm_info[save_col] = conservation_score
        if return_score:
            if unique_isoforms:
                num_isoforms = alt_ptms['Isoform ID'].nunique()
            else:
                num_isoforms = alt_ptms['Alternative Transcript'].nunique()
            return self.ptm_info[self.ptm_info[save_col] == 1].shape[0]/self.ptm_info.shape[0], num_isoforms
        
    def addSpliceEventsToAlternative(self, splice_events_df):
        splice_events_df = splice_events_df[['Exon ID (Canonical)', 'Exon ID (Alternative)', 'Alternative Transcript', 'Event Type']].drop_duplicates()
        alternative_ptms = self.alternative_ptms.copy()
        alternative_ptms = alternative_ptms.drop('Exon ID (Alternative)', axis = 1)
        alternative_ptms['Exon ID (Canonical)'] = alternative_ptms['Exon ID (Canonical)'].apply(lambda x: x.split(';'))
        alternative_ptms = alternative_ptms.explode('Exon ID (Canonical)').drop_duplicates()
        alternative_ptms = alternative_ptms.merge(splice_events_df, on = ['Exon ID (Canonical)', 'Alternative Transcript'], how = 'left')
        
        
        exploded_ptms = self.explode_PTMinfo()
        #check PTMs in mutually exclusive exons to see if they might be conserved
        mxe_ptm_candidates = alternative_ptms[alternative_ptms['Event Type'] == 'Mutually Exclusive'].copy()
        alternative_ptms = alternative_ptms[alternative_ptms['Event Type'] != 'Mutually Exclusive']
        alt_residue_list = []
        alt_position_list = []
        for i, row in mxe_ptm_candidates.iterrows():
            if not utility.stringToBoolean(row['Ragged']):
                ptm = row['Source of PTM']
                
                #get canonical exon info
                canonical_exon_id = row['Exon ID (Canonical)']
                ptm_info_of_interest = exploded_ptms.loc[(exploded_ptms['PTM'] == ptm) & (exploded_ptms['Exon stable ID'] == canonical_exon_id)].squeeze()
                canonical_exon = self.exons[(self.exons['Exon stable ID'] == canonical_exon_id) & (self.exons['Transcript stable ID'] == ptm_info_of_interest['Transcripts'])].squeeze()
                canonical_exon_sequence = Seq(canonical_exon['Exon AA Seq (Full Codon)'])
                
                #get alternative_exon info
                alternative_exon_id = row['Exon ID (Alternative)']
                alternative_exon = self.exons[(self.exons['Exon stable ID'] == alternative_exon_id) & (self.exons['Transcript stable ID'] == row['Alternative Transcript'])].squeeze()
                alternative_exon_sequence = Seq(alternative_exon['Exon AA Seq (Full Codon)'])
                
                #align sequences
                alignment = returnAlignment(canonical_exon_sequence, alternative_exon_sequence, canonical_exon_id, alternative_exon_id)
                
                #find location of PTM in canonical exon 
                canonical_gap_map = getGapMaps(alignment, canonical_exon_id)
                ptm_pos_in_exon = int(float(ptm_info_of_interest['Exon Location (AA)'])) - 1
                ptm_pos_in_alignment = canonical_gap_map[ptm_pos_in_exon]
                
                #look for ptm in alternative sequence, if present, find location of ptm in alternative isoform
                alternative_gap_map = getGapMaps(alignment, alternative_exon_id, reverse = True)
                if ptm_pos_in_alignment in alternative_gap_map.keys():
                    isoform_sequence = self.transcripts.loc[row['Alternative Transcript'], 'Amino Acid Sequence']
                    ptm_pos_in_alt_exon = alternative_gap_map[ptm_pos_in_alignment]
                    ptm_pos_in_isoform = int(np.ceil(float(alternative_exon['Exon Start (Protein)'])))+ptm_pos_in_alt_exon+1
                    alternative_residue = alternative_exon_sequence[ptm_pos_in_alt_exon]
                    
                    if alternative_residue == row['Canonical Residue']:
                        alt_residue_list.append(alternative_residue)
                        alt_position_list.append(ptm_pos_in_isoform)
                    else:
                        alt_residue_list.append(np.nan)
                        alt_position_list.append(np.nan)
                else:
                    alt_residue_list.append(np.nan)
                    alt_position_list.append(np.nan)
            else:
                alt_position_list.append(np.nan)
                alt_residue_list.append(np.nan)
                
        mxe_ptm_candidates['Alternative Residue'] = alt_residue_list
        mxe_ptm_candidates['Alternative Protein Position (AA)'] = alt_position_list
        
        #replace mxe ptm candidates with new info
        mxe_ptm_candidates['Mapping Result'] = mxe_ptm_candidates.apply(lambda x: 'Success' if x['Alternative Residue'] == x['Alternative Residue'] else 'Not Found', axis = 1)
        alternative_ptms = pd.concat([alternative_ptms, mxe_ptm_candidates])
        
        #collapse into rows for matching event types and exon ids
        cols = [col for col in alternative_ptms.columns if col != 'Event Type' and col != 'Source Exons']
        alternative_ptms = alternative_ptms.replace(np.nan, 'nan')
        alternative_ptms = alternative_ptms.groupby(cols)['Event Type'].agg(';'.join).reset_index()
        alternative_ptms = alternative_ptms.replace('nan', np.nan)
        
        #save to mapper object
        self.alternative_ptms = alternative_ptms.copy()
    
    def annotateAlternativePTMs(self):
        #self.alternative_ptms['Canonical PTM'] = self.alternative_ptms['Protein'] + '_' + self.alternative_ptms['Residue'] + self.alternative_ptms['Canonical Protein Location (AA)'].astype(str)
        if 'TRIFID Score' in self.transcripts.columns and 'TRIFID Score' not in self.alternative_ptms.columns:
            self.alternative_ptms = self.alternative_ptms.merge(self.transcripts['TRIFID Score'], right_index = True, left_on = 'Alternative Transcript', how = 'left')
            
        if os.path.exists(config.processed_data_dir + 'splice_events.csv') and 'Event Type' not in self.alternative_ptms.columns:
            print('Adding splice events to alternative dataframes and checking MXE events for conserved PTMs')
            sevents = pd.read_csv(config.processed_data_dir + 'splice_events.csv')
            self.addSpliceEventsToAlternative(sevents)
            

        print('Getting flanking sequences around PTMs in alternative isoforms')
        flank = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Position (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                flank.append(np.nan)
            else:
                flank.append(self.getFlankingSeq(int(pos), transcript_id, flank_size = 10))
        self.alternative_ptms['Flanking Sequence'] = flank
        

        print('Getting tryptic fragments that include each PTM in alternative isoforms')
        tryptic = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Position (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                tryptic.append(np.nan)
            else:
                tryptic.append(self.getTrypticFragment(int(pos), transcript_id))
        self.alternative_ptms['Tryptic Fragment'] = tryptic

            

    def getIsoformSpecificPTMs(self, required_length = 20):
        """
        Reduce alternative ptm dataframe to ptms that are unique to a specific protein sequence, rather than a specific transcript
        """
        isoforms = self.isoforms[['Isoform ID', 'Transcript stable ID', 'Isoform Length']].copy()
        isoforms['Transcript stable ID'] = isoforms['Transcript stable ID'].apply(lambda x: x.split(';'))
        isoforms = isoforms.explode('Transcript stable ID')
        isoforms = isoforms.rename({'Transcript stable ID': 'Alternative Transcript'}, axis = 1)
        isoform_ptms = self.alternative_ptms.merge(isoforms, on = 'Alternative Transcript', how = 'left').copy()
        isoform_ptms = isoform_ptms[isoform_ptms['Isoform Length'] >= required_length]
        isoform_ptms = isoform_ptms[['Isoform ID', 'Source of PTM', 'Frame', 'Alternative Protein Position (AA)', 'Alternative Residue', 'Modifications','Flanking Sequence', 'Tryptic Fragment']]
        isoform_ptms = isoform_ptms.drop_duplicates()
        self.isoform_ptms = isoform_ptms
        
    def load_PTMmapper(self):
        #check if each data file exists: if it does, load into mapper object
        if os.path.exists(config.processed_data_dir + 'exons.csv'):
            print('Loading exon-specific data')
            self.exons = pd.read_csv(config.processed_data_dir + 'exons.csv')
        else:
            self.exons = None
            
        if os.path.exists(config.processed_data_dir + 'transcripts.csv'):
            print('Loading transcript-specific data')
            self.transcripts = pd.read_csv(config.processed_data_dir + 'transcripts.csv', index_col = 0)
        else:
            self.transcripts = None
        
        if os.path.exists(config.processed_data_dir + 'genes.csv'):
            print('Loading gene-specific info')
            self.genes = pd.read_csv(config.processed_data_dir + 'genes.csv', index_col = 0)
        else:
            self.genes = None
            
        if os.path.exists(config.processed_data_dir + 'isoforms.csv'):
            print('Loading unique protein isoforms')
            self.isoforms = pd.read_csv(config.processed_data_dir + 'isoforms.csv')
        else:
            self.isoforms = None
            
        if os.path.exists(config.processed_data_dir + 'proteins.csv'):
            print('Loading protein-specific info')
            self.proteins = pd.read_csv(config.processed_data_dir + 'proteins.csv', index_col = 0)
        else:
            self.proteins = None
            
        if os.path.exists(config.processed_data_dir + 'ptm_info.csv'):
            print('Loading information on PTMs on canonical proteins')
            self.ptm_info = pd.read_csv(config.processed_data_dir + 'ptm_info.csv',index_col = 0)
            
            #check to see if ptm_info is collapsed or not (if each row is a unique ptm)
            if len(np.unique(self.ptm_info.index)) != self.ptm_info.shape[0]:
                self.ptm_info = self.ptm_info.reset_index()
                self.ptm_info = self.ptm_info.rename({'index':'PTM'}, axis = 1)
        else:
            self.ptm_info = None
            
        if os.path.exists(config.processed_data_dir + 'ptm_coordinates.csv'):
            print('Loading information on PTMs on canonical proteins')
            self.ptm_coordinates = pd.read_csv(config.processed_data_dir + 'ptm_coordinates.csv',index_col = 0,
                                                dtype = {'Source of PTM': str, 'Chromosome/scaffold name': str, 'Gene Location':int,
                                                'Ragged': str})
        else:
            self.ptm_coordinates = None
            
        if os.path.exists(config.processed_data_dir + 'alternative_ptms.csv'):
            print('Loading information on PTMs on alternative proteins')
            self.alternative_ptms = pd.read_csv(config.processed_data_dir + 'alternative_ptms.csv', dtype = {'Exon ID (Alternative)':str, 'Chromosome/scaffold name': str, 'Ragged':str, 'Genomic Coordinates':str, 'Second Exon': str, 'Alternative Residue': str, 'Protein':str})
        else:
            self.alternative_ptms = None


    
def RNA_to_Prot(pos, cds_start):
    """
    Given nucleotide location, indicate the location of residue in protein
    """
    return (int(pos) - int(cds_start)+3)/3

def Prot_to_RNA(pos, cds_start):
    """
    Given residue location, indicate the location of first nucleotide in transcript (first nucleotide in the codon)
    """
    return (int(pos)*3 -3) + int(cds_start)
    
def getDistanceToBoundary(exon, gene_loc, strand):
    if strand == 1:
        distance_to_bound = exon['Exon End (Gene)'] - gene_loc
    else:
        distance_to_bound = gene_loc - exon['Exon Start (Gene)']
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
    
def getPTMExonRank(row):
    """
    Given a single ptm instance from ptm_info, determine the rank of the exon that the PTM is located in. Assumes exon cut information has been added and it has already been split into list. Used by MapPTMs_all() function.
    """
    normed_cuts = row['Transcript Location (NC)'] - row['Exon cuts'] 
    for c in range(len(normed_cuts)):
        if normed_cuts[c] < 0:
            return c + 1
    
def getPTMLoc(exon, mapper, gene_loc, strand):
    #make sure transcript associated with exon contains coding information
    transcript = mapper.transcripts.loc[exon['Transcript stable ID']]
    if transcript['Relative CDS Start (bp)'] != 'No coding sequence' and transcript['Relative CDS Start (bp)'] != 'error:no match found':
        #check frame: if in frame, return residue and position
        frame, residue, aa_pos = utility.checkFrame(exon, transcript, gene_loc, 
                                 loc_type = 'Gene', strand = strand, return_residue = True)
        return frame, residue, aa_pos
    else:
        return np.nan, np.nan, np.nan

def convertToHG19(hg38_location, chromosome, strand, liftover_object = None):
    """
    Use pyliftover to convert from hg38 to hg19 coordinates systems
    
    Parameters
    ----------
    hg38_location: int
        genomic location to convert, in hg38 version of Ensembl
        
    Returns
    -------
    hg19_coordinates: int
        genomic coordinates in hg19 version of Ensembl
    """
    if liftover_object is None:
        liftover_object = pyliftover('hg19','hg38')
    
    if strand == 1:
        strand = '+'
    else:
        strand = '-'
    
    chromosome = f'chr{chromosome}'
    results = liftover_object.convert_coordinate(chromosome, hg38_location - 1, strand)
    if len(results) > 0:
        new_chromosome = results[0][0]
        new_strand = results[0][2]
        if new_chromosome == chromosome and new_strand == strand:
            return results[0][1] + 1
        else:
            return -1
    else:
        return np.nan
        
        
def returnAlignment(seq1, seq2, canonical_exon_id, alternative_exon_id):
    """
    Given two exon sequences in amino acids (usually two exons thought to be mutually exclusive), align the two sequences and return aligned object
    
    Parameters
    ----------
    seq1: str
        amino acid sequence associated with canonical exon
    seq2: str
        amino acid sequence associated with alternative exon
    canonical_exon_id: str
        Ensembl id associated with canonical exon
    alternative_exon_id: str
        Ensembl id associated with alternative exon
        
    Returns
    -------
    aln: Cogent3 alignement object
        alignment of the two exon sequences
    """
    #align (first gap has -10 penalty, subsequent have -2)
    alignments = pairwise2.align.globalxs(seq1, seq2, -10, -2)
    
    #extract alignment sequences
    aln_seq1 = alignments[0][0]
    aln_seq2 = alignments[0][1]

    #use cogent
    aln = make_aligned_seqs([[canonical_exon_id,aln_seq1], [alternative_exon_id, aln_seq2]], array_align=True, moltype ='protein')
    return aln
    
def getGapMaps(aln, exon_id, reverse = False):
    """
    Given an alignment object, return the gap maps for each sequence in the alignment
    
    Parameters
    ----------
    aln: Cogent3 alignment object, possibly from returnAlignment function()
        alignment of two sequences
    exon_id: str
        exon_id for which to get the gap map
    
    Returns
    -------
    alignment_map: dict
        dictionary indicating the location of each residue in the canonical sequence in the gapped alignment
    """
    #get gapped sequence
    gap_alignment = aln.get_gapped_seq(exon_id)
    if reverse:
        align_map = gap_alignment.gap_maps()[1]
    else:
        align_map = gap_alignment.gap_maps()[0]
    return align_map
    

def run_mapping(restart_all = False, restart_mapping = False, exon_sequences_fname = 'exon_sequences.fasta.gz',
                coding_sequences_fname = 'coding_sequences.fasta.gz', trifid_fname = 'APPRIS_functionalscores.txt'):
    mapper = PTM_mapper()
    
    if (mapper.exons is None and mapper.transcripts is None and mapper.genes is None) or restart_all:
        mapper.genes, mapper.transcripts, mapper.exons = processing.downloadMetaInformation()
        
    if 'Exon Sequence' not in mapper.exons.columns or restart_all:
        # Process_exons and transcripts
        print('Processing exon sequences and adding to meta information')
        #load exon sequences
        if os.path.exists(config.source_data_dir+exon_sequences_fname):
            exon_sequences = utility.processEnsemblFasta(config.source_data_dir+exon_sequences_fname, id_col = 'Exon stable ID', seq_col = 'Exon Sequence')
            mapper.exons = processing.processExons(mapper.exons, exon_sequences)
            mapper.exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)
        else:
            raise ValueError(f'{exon_sequences_fname} not found in the source data directory. Make sure file is located in the source data directory and matches the directory and file name')
    
    if 'Transcript Sequence' not in mapper.transcripts.columns or restart_all:
        print('Extracting transcript sequence information')
        #load coding sequences
        if os.path.exists(config.source_data_dir+coding_sequences_fname):
            coding_sequences = utility.processEnsemblFasta(config.source_data_dir+coding_sequences_fname, id_col = 'Transcript stable ID', seq_col = 'Coding Sequence')
            coding_sequences.index = coding_sequences['Transcript stable ID']
            coding_sequences = coding_sequences.drop('Transcript stable ID', axis = 1)
        else:
            raise ValueError(f'{coding_sequences_fname} not found in the source data directory. Make sure file is located in the source data directory and matches the directory and file name')
        #load trifid scores
        if trifid_fname is None:
            mapper.transcripts = processing.processTranscripts(mapper.transcripts, coding_sequences, mapper.exons)
        elif os.path.exists(config.source_data_dir+trifid_fname):
            TRIFID = pd.read_csv(config.source_data_dir + trifid_fname,sep = '\t')
            #process transcript data
            mapper.transcripts = processing.processTranscripts(mapper.transcripts, coding_sequences, mapper.exons, APPRIS = TRIFID)
        else:
            print('Note: TRIFID functional score file not found. Proceeding without adding to transcripts dataframe')
            mapper.transcripts = processing.processTranscripts(mapper.transcripts, coding_sequences, mapper.exons)
        print('saving\n')
        mapper.transcripts.to_csv(config.processed_data_dir + 'transcripts.csv')
            
    if 'Exon AA Seq (Full Codon)' not in mapper.exons.columns or restart_all:
        print('Getting exon-specific amino acid sequence\n')
        mapper.exons = processing.getAllExonSequences(mapper.exons, mapper.transcripts, mapper.genes)
        mapper.exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)
        
    if config.available_transcripts is None:
        print('Identifying transcripts with matching information from UniProt canonical proteins in ProteomeScout')
        processing.getMatchedTranscripts(mapper.transcripts, update = restart_all)
    
    if mapper.proteins is None or restart_all:
        print('Getting protein-specific information')
        mapper.proteins = processing.getProteinInfo(mapper.transcripts, mapper.genes)
        mapper.proteins.to_csv(config.processed_data_dir + 'proteins.csv')
        
    if mapper.isoforms is None or restart_all:
        print('Getting unique protein isoforms from transcript data')
        mapper.isoforms = processing.getIsoformInfo(mapper.transcripts)
        mapper.isoforms.to_csv(config.processed_data_dir + 'isoforms.csv', index = False)

    if restart_all:
        restart_mapping = True
        
    if mapper.ptm_info is None or restart_mapping:
        mapper.findAllPTMs(collapse = False)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        
    if mapper.ptm_coordinates is None or restart_mapping:
        print('Mapping PTMs to their genomic location')
        mapper.mapPTMs_all(restart = restart_mapping)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        mapper.ptm_coordinates.to_csv(config.processed_data_dir + 'ptm_coordinates.csv')

    
    ####run additional analysis
    if 'Tryptic Fragment' not in mapper.ptm_info.columns or restart_mapping:
        mapper.getAllTrypticFragments()
    if 'Flanking Sequence' not in mapper.ptm_info.columns or restart_mapping:
        mapper.getAllFlankingSeqs(flank_size = 10)
    if 'inDomain' not in mapper.ptm_info.columns or restart_mapping:
        mapper.findAllinDomains()
    print('saving\n')
    mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
    
    if mapper.alternative_ptms is None:
        print('Mapping PTM sites onto alternative isoforms')
        mapper.mapPTMsToAlternativeExons()
        mapper.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv', index = False)
        print('saving\n')
   
        
    if not os.path.exists(config.processed_data_dir + 'splice_events.csv'):
        print('Identify splice events that result in alternative isoforms')
        splice_events_df = get_splice_events.identifySpliceEvents_All(mapper.exons, mapper.proteins, mapper.transcripts, mapper.genes)
        print('saving\n')
        splice_events_df.to_csv(config.processed_data_dir + 'splice_events.csv', index = False)
        
    if 'Tryptic Fragment' not in mapper.alternative_ptms.columns:
        print('Annotate PTM sites on alternative isoforms')
        mapper.annotateAlternativePTMs()
        print('saving\n')
        mapper.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv', index = False)
        
    if 'Number of Conserved Transcripts' not in mapper.ptm_info.columns:
        print('Calculating conservation of each PTM across alternative transcripts')
        mapper.calculate_PTMconservation()
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        
    return mapper