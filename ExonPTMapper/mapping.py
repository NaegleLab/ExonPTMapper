import pandas as pd
import numpy as np
from Bio import SeqIO
import os
import gzip
import re
import sys
import time
import multiprocessing
from tqdm import tqdm
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
            ptm_df.index = ptm_df['Protein']+'_'+ptm_df['Residue']+ptm_df['PTM Location (AA)']
       
        return ptm_df
        
    def findAllPTMs(self, PROCESSES = 1):
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
            self.ptm_info.index = self.ptm_info['Protein']+'_'+self.ptm_info['Residue']+self.ptm_info['PTM Location (AA)'].astype(int).astype(str)
            #collapse rows with duplicate indexes, but different transcripts or modififications into a single row, with each transcript or modification seperated by comma
            grouped_genes = self.ptm_info.groupby(level = 0)['Gene'].apply(','.join)
            self.ptm_info['Genes'] = grouped_genes
            grouped_transcripts = self.ptm_info.groupby(level = 0)['Transcript'].apply(','.join)
            self.ptm_info['Transcripts'] = grouped_transcripts
            grouped_mods = self.ptm_info.groupby(level = 0)['Modification'].apply(set).apply(';'.join)
            self.ptm_info['Modifications'] = grouped_mods
            self.ptm_info = self.ptm_info.drop(['Gene','Transcript','Modification'], axis = 1)
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
                        index = ['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged'],
                        name = ptm)
                        
        
    def mapPTMs_all(self, save_iter = 5000, restart = False, PROCESSES = 1):
        """
        For all ptms in ptm_info, map to genome and save in ptm_positions 
        """
        #process ptms to operate on each unique PTM instance separately
        
        #exploded_ptms = self.explode_PTMinfo()
        
        #check to make sure datatypes are correct
        if self.ptm_info['PTM Location (AA)'].dtypes != int:
            self.ptm_info['PTM Location (AA)'] = self.ptm_info['PTM Location (AA)'].astype(int)
            
        if PROCESSES > 1:
            raise ValueError('Multiprocessing is not yet ready')
            #get ptms and split into separate lists
            ptm_list = self.ptm_info.index.values
            pool = multiprocessing.Pool(processes = PROCESSES)
            results = tqdm(pool.map(self.mapPTM, ptm_list))

        else:
            results = []
            iteration = 1
            #check if mapping has been run previously
            if 'Exon stable ID' in self.ptm_info.columns and not restart:
                print('Found data from previous runs. Only analyzing rows without exon information. If you would like to analyze all ptms, set restart = True.\n')
                #get ptms that still need to be analyzed and those that already have been
                ptms_for_analysis = self.ptm_info[self.ptm_info['Exon stable ID'].isna()].index
                analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged']]
                num_analyzed_ptms = analyzed_ptms.shape[0]
                iteration = iteration+num_analyzed_ptms
                print(f'Analysis restarting at row {iteration}\n')
                for ptm in tqdm(ptms_for_analysis, desc = 'Mapping PTMs to exon and gene'):
                    results.append(self.mapPTM(ptm))
                    
                    #if save_iter has been reached, temporarily save results
                    if iteration % save_iter == 0:
                        #print(f'Saving results from iteration {iteration}\n')
                        tmp_results = pd.concat(results, axis = 1).T
                        tmp_results = pd.concat([analyzed_ptms, tmp_results])
                        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged'], axis = 1)
                        self.ptm_info = self.ptm_info.join(tmp_results)
                        
                        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
                        
                    iteration += 1
            else:
                for ptm in tqdm(self.ptm_info.index, desc = 'Mapping PTMs to exon and gene'):
                    results.append(self.mapPTM(ptm))
                    
                    #if save_iter has been reached, temporarily save results
                    if iteration % save_iter == 0:
                        #print(f'Saving results from iteration {iteration}')
                        tmp_results = pd.concat(results, axis = 1).T
                        if 'Exon stable ID' in self.ptm_info.columns:
                            self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)','Distance to C-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged'], axis = 1)
                        self.ptm_info = self.ptm_info.join(tmp_results)
                        
                        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
                    iteration += 1

        #concatenate series into dataframe
        results = pd.concat(results, axis = 1).T

        analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank','Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)','Distance to Closest Boundary (NC)', 'Ragged']]
        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank','Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)','Distance to Closest Boundary (NC)', 'Ragged'], axis = 1)
        results = pd.concat([analyzed_ptms, results])
        #add to ptm_info dataframe
        self.ptm_info = self.ptm_info.join(results) 

        self.ptm_info = self.ptm_info.drop_duplicates()
        print('All data analyzed: saving full ptm_info results')
        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')

            
    def explode_PTMinfo(self, explode_cols = ['Genes', 'Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)','Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to Closest Boundary (NC)', 'Ragged']):
        exploded_ptms = self.ptm_info.copy()
        #check columns that exist
        explode_cols = [col for col in explode_cols if col in exploded_ptms.columns.values]
        #split different entries
        for col in explode_cols:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(','))
      
        exploded_ptms = exploded_ptms.explode(explode_cols)
        exploded_ptms = exploded_ptms.reset_index()
        exploded_ptms = exploded_ptms.rename({'index':'PTM'}, axis = 1)
        return exploded_ptms
        
    def collapse_PTMinfo(self, all_cols = ['Genes', 'Transcripts','Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)','Exon stable ID', 'Exon Rank'], unique_cols = ['Modifications', 'Exon Location (AA)', 'Protein', 'PTM Location (AA)', 'Residue']):
        ptms = self.ptm_info.copy()
        ptms = ptms.astype(str)
        grouped_mods = ptms.groupby(level = 0)
        collapsed_ptms = []
        for col in self.ptm_info.columns:
            if col in unique_cols:
                collapsed_ptms.append(grouped_mods[col].apply(set).apply(','.join))
            else:
                collapsed_ptms.append(grouped_mods[col].apply(','.join))
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
            if ',' in transcript:
                transcript = transcript.split(',')[0]
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
            if ',' in transcript:
                transcript = transcript.split(',')[0]
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
        #check to make sure gene location is str so that multiple locs can be split into list
        if not isinstance(ptm['Gene Location (NC)'], str):
            ptm['Gene Location (NC)'] = str(ptm['Gene Location (NC)'])
            
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
        
        results = None
        #isolate info into unique genes (if multiple associated with protein/ptm of interest)
        gene_locs = ptm['Gene Location (NC)'].split(',')
        exons = ptm['Exon stable ID'].split(',')
        genes = self.genes.loc[ptm['Genes'].split(',')]
        exon_rank = ptm['Exon Rank'].split(',')
        transcript_ids = ptm['Transcripts'].split(',')
        ragged = ptm['Ragged'].split(',')
        for g in range(len(gene_locs)):
            gene = genes.iloc[g]
            source_exon_id = exons[g]
            gene_loc = gene_locs[g]
            if gene_loc != 'Exons Not Found' and gene_loc != 'CDS fail':
                gene_loc = int(gene_loc)
                strand = gene['Strand']
                #grab all exons containing ptm site (based on genome location and ensuring they come from same gene info)
                ptm_exons = trim_exons[(trim_exons['Exon Start (Gene)'] <= gene_loc) & (trim_exons['Exon End (Gene)'] >= gene_loc) & (trim_exons['Gene stable ID'] == gene.name)].copy()
                
                #if site is ragged, check second exon for conserved region
                #if utility.stringToBoolean(ragged[0]):
                #    first_contributing_exon_rank = exon_rank[g]
                #    second_contributing_exon_rank = first_contributing_exon_rank + 1
                #    other_contributing_exon = exons[(exons['Transcript stable ID'] == transcripts[g]) & (exons['Exon rank in transcript'] == second_contributing_exon_rank)].squeeze()
                #    if strand == 1:
                #        loc_other_exon_in_gene = other_contributing_exon['Exon Start (Gene)']
                #    else:
                #        loc_other_exon_in_gene = other_contributing_exon['Exon End (Gene)']
                #    #check if second exon is conserved AND was not already found when looking for first exon
                #    second_ptm_exons = trim_exons[(trim_exons['Exon Start (Gene)'] <= loc_other_exon_in_gene) & (trim_exons['Exon End (Gene)'] >= loc_other_exon_in_gene) & (trim_exons['Gene stable ID'] == gene.name) & ~(trim_exons['Transcript stable ID'].isin(ptm_exons['Transcript stable ID']))].copy()
                #else:
                #    second_ptm_exons = None
                    
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
                        ptm_loc_info = ptm_exons.apply(getPTMLoc, args = (self, gene_loc, strand), axis = 1)
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
                        ptm_exons['Residue'] = residue_list
                        ptm_exons['Position'] = aa_pos_list
                        ptm_exons['Second Exon'] = second_exon_list
                        
                #if second_ptm_exons is not None:
                #    coordinates = []
                #    frame_list = []
                #    residue_list = []
                #    position_list = []
                #    for i, row in second_ptm_exons.iterrows():
                #        #get PTM distance to boundary (should be the same for all canonical transcripts). this will indicate how much each exon contributes to ragged site
                #        dist_to_boundary =  int(ptm['Distance to Closest Boundary (NC)'].split(',')[0])
                #        if strand == 1:
                #            start_second_exon = row['Exon Start (Transcript)']
                #            codon_start = start_second_exon - (3+dist_to_boundary)
                #        else:
                #            codon_start = row['Exon End (Transcript)']
                #        
                        #check frame
                #        transcript = mapper.transcripts.loc[transcript_ids[g]]
                #        transcript_sequence = transcripts.loc[transcripts[g], 'Transcript Sequence']
                #        codon = transcript_sequence[codon_start:codon_start+3]
                #        new_residue = utility.codon_dict[codon]
                #        residue_list.append(new_residue)
                #        if new_residue == canonical_ptm['Residue']:
                #            position_list.append((codon_start - transcript['Relative CDS Start (bp)'] + 3)/3)
                            
                    #second_ptm_exons['Genomic Coordinates'] =
                    #second_ptm_exons['Frame'] =
                    #second_ptm_exons['Residue']
                    #second_ptm_exons['Position']
                if ptm_exons.shape[0] > 0:
                    if results is None:
                        results = ptm_exons.copy()
                    else:
                        results = pd.concat([results, ptm_exons])
                    
        return results
        
    def mapPTMsToAlternativeExons(self):
        #get all alternative transcripts with available coding info
        available_transcripts = self.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
        alternative_transcripts = config.translator.loc[config.translator['Uniprot Canonical'] != 'Canonical', 'Transcript stable ID']
        available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
        
        #grab exons associated with available transcripts
        trim_exons = self.exons[self.exons['Transcript stable ID'].isin(available_transcripts)]
        trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()
        
        #convert gene location into string
        self.ptm_info['Gene Location (NC)'] = self.ptm_info['Gene Location (NC)'].astype(str)
        
        alt_ptms = None
        for i,ptm in tqdm(self.ptm_info.iterrows()):
            ptm_exons = self.mapPTMtoExons(ptm, trim_exons = trim_exons)
            if ptm_exons is not None:
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
        alternative = self.proteins.dropna(subset = 'Alternative Transcripts (All)').copy()
        alternative['Alternative Transcripts (All)'] = alternative['Alternative Transcripts (All)'].apply(lambda x: x.split(','))
        alternative = alternative.explode('Alternative Transcripts (All)').reset_index()
        alternative = alternative[['UniProtKB/Swiss-Prot ID', 'Alternative Transcripts (All)']]
        alternative = alternative[alternative['Alternative Transcripts (All)'].isin(available_transcripts)]
        ptms = self.ptm_info.reset_index()[['index', 'Protein']].drop_duplicates()
        ptms = ptms.rename({'index':'PTM'}, axis = 1)
        alternative = alternative.merge(ptms, on = 'Protein')
        #identify PTMs that were found in alternative transcripts
        #missing = ~alternative[['Transcript stable ID', 'PTM']].isin(alt_ptms[['Transcript stable ID', 'PTM']])
        #missing_alternative = alternative[missing]

        
        alt_ptms = alt_ptms.merge(alternative, on = ['Transcript stable ID', 'PTM'], how = 'outer')
        alt_ptms = alt_ptms.rename({'Alternative Transcripts (All)': 'Transcript stable ID', 'UniProtKB/Swiss-Prot ID':'Protein', 'Position': 'Alternative Protein Location (AA)', 
'Residue':'Alternative Residue'}, axis = 1)
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
        
        self.alternative_ptms = alt_ptms
    
    def calculate_PTMconservation(self):
        if self.alternative_ptms is None:
            raise AttributeError('No alternative_ptms attribute. Must first map ptms to alternative transcripts with mapPTMsToAlternative()')
        else:
            #conserved transcripts
            conserved_transcripts = self.alternative_ptms[self.alternative_ptms['Mapping Result'] == 'Success'].groupby('PTM')['Alternative Transcript'].apply(list)
            num_conserved_transcripts = conserved_transcripts.apply(len)
            conserved_transcripts = conserved_transcripts.apply(','.join)
            
            #lost transcripts
            lost_transcripts = self.alternative_ptms[self.alternative_ptms['Mapping Result'] != 'Success'].groupby('PTM')['Alternative Transcript'].apply(list)
            num_lost_transcripts = lost_transcripts.apply(len)
            lost_transcripts = lost_transcripts.apply(','.join)
            
            #save results in ptm_info
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
            self.ptm_info['PTM Conservation Score'] = conservation_score
        
    def addSpliceEventsToAlternative(self, splice_events_df):
        splice_events_df = splice_events_df[['Exon ID (Canonical)', 'Alternative Transcript', 'Event Type']]
        self.alternative_ptms = self.alternative_ptms.merge(splice_events_df, on = ['Exon ID (Canonical)', 'Alternative Transcript'], how = 'left')
    
    def annotateAlternativePTMs(self):
        #self.alternative_ptms['Canonical PTM'] = self.alternative_ptms['Protein'] + '_' + self.alternative_ptms['Residue'] + self.alternative_ptms['Canonical Protein Location (AA)'].astype(str)
        if 'TRIFID Score' in self.transcripts.columns:
            self.alternative_ptms = self.alternative_ptms.merge(self.transcripts['TRIFID Score'], right_index = True, left_on = 'Alternative Transcript', how = 'left')
            
        
        print('Getting flanking sequences around PTMs in alternative isoforms')
        flank = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Location (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                flank.append(np.nan)
            else:
                flank.append(self.getFlankingSeq(int(pos), transcript_id, flank_size = 10))
        self.alternative_ptms['Flanking Sequence'] = flank
        
        print('Getting tryptic fragments that include each PTM in alternative isoforms')
        tryptic = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Location (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                tryptic.append(np.nan)
            else:
                tryptic.append(self.getTrypticFragment(int(pos), transcript_id))
        self.alternative_ptms['Tryptic Fragment'] = tryptic
        
        if os.path.exists(config.processed_data_dir + 'splice_events.csv'):
            sevents = pd.read_csv(config.processed_data_dir + 'splice_events.csv')
            self.addSpliceEventsToAlternative(sevents)
            
        self.alternative_ptms = self.alternative_ptms.merge(self.transcripts['TRIFID Score'], right_index = True, left_on = 'Alternative Transcript', how = 'left')
            
        
        
            
        
    def savePTMs(self):
        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')

        
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
            
        if os.path.exists(config.processed_data_dir + 'proteins.csv'):
            print('Loading protein-specific info')
            self.proteins = pd.read_csv(config.processed_data_dir + 'proteins.csv', index_col = 0)
        else:
            self.proteins = None
            
        if os.path.exists(config.processed_data_dir + 'ptm_info.csv'):
            print('Loading information on PTMs on canonical proteins')
            self.ptm_info = pd.read_csv(config.processed_data_dir + 'ptm_info.csv',index_col = 0)
        else:
            self.ptm_info = None
            
        if os.path.exists(config.processed_data_dir + 'alternative_ptms.csv'):
            print('Loading information on PTMs on alternative proteins')
            self.alternative_ptms = pd.read_csv(config.processed_data_dir + 'alternative_ptms.csv')
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
            coding_sequences = utility.processEnsemblFasta(config.source_data_dir+'coding_sequences.fasta.gz', id_col = 'Transcript stable ID', seq_col = 'Coding Sequence')
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
        
    print('Identifying transcripts with matching information from UniProt canonical proteins in ProteomeScout')
    processing.getMatchedTranscripts(mapper.transcripts, update = restart_all)
    
    if mapper.proteins is None or restart_all:
        print('Getting protein-specific information')
        mapper.proteins = processing.getProteinInfo(mapper.transcripts, mapper.genes)
        mapper.proteins.to_csv(config.processed_data_dir + 'proteins.csv')
        
    print('Saving data files\n')
    mapper.exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)
    mapper.transcripts.to_csv(config.processed_data_dir + 'transcripts.csv')

    if restart_all:
        restart_mapping = True
        
    if mapper.ptm_info is None or restart_mapping:
        mapper.findAllPTMs()
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        
    if 'Gene Location (NC)' not in mapper.ptm_info.columns:
        mapper.mapPTMs_all(restart = restart_mapping)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
    elif mapper.ptm_info['Gene Location (NC)'].isna().any():
        mapper.mapPTMs_all(restart = restart_mapping)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
    
    ####run additional analysis
    if 'Tryptic Fragment' not in mapper.ptm_info.columns or restart_mapping:
        mapper.getAllTrypticFragments()
    if 'Flanking Sequence' not in mapper.ptm_info.columns or restart_mapping:
        mapper.getAllFlankingSeqs()
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
    
