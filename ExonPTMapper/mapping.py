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
                trans = trans
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
            return np.repeat('CDS fail', 8)
        elif self.exons[self.exons['Transcript stable ID'] == tid].shape[0] == 0:
            return np.repeat('Exons Not Found', 8)
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
            
            #calculate distance to boundary
            nterm_distance = str(PTM_start - exon_of_interest['Exon Start (Transcript)'])
            cterm_distance = str(exon_of_interest['Exon End (Transcript)'] - (PTM_start + 3))
    
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
                
            return gene_codon_start, str(PTM_start), str(exon_codon_start), exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance
        
        
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
            for t in transcript_ids:
                map_results = self.mapPTM_singleTranscript(position, t)
                gene_codon_start.append(map_results[0])
                PTM_start.append(map_results[1])
                exon_codon_start.append(map_results[2])
                exon_aa_start.append(map_results[3])
                exon_id.append(map_results[4])
                exon_rank.append(map_results[5])
                nterm_distance.append(map_results[6])
                cterm_distance.append(map_results[7])
                
            #convert lists to strings
            PTM_start = ','.join(PTM_start)   
            exon_id = ','.join(exon_id) 
            exon_rank = ','.join(exon_rank)
            nterm_distance = ','.join(nterm_distance)
            cterm_distance = ','.join(cterm_distance)
            exon_codon_start = ','.join(exon_codon_start)
            gene_codon_start = ','.join(gene_codon_start)
            exon_aa_start = ','.join(exon_aa_start)
        else:
            map_results = self.mapPTM_singleTranscript(position, transcript_ids[0])
            gene_codon_start, PTM_start, exon_codon_start, exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance = map_results

        
        return pd.Series(data = [gene_codon_start, PTM_start, exon_codon_start, exon_aa_start, exon_id, exon_rank, nterm_distance, cterm_distance],
                        index = ['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)'],
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
                analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)']]
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
                        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)'], axis = 1)
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
                            self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank', 'Distance to N-terminal Splice Boundary (NC)','Distance to C-terminal Splice Boundary (NC)'], axis = 1)
                        self.ptm_info = self.ptm_info.join(tmp_results)
                        
                        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
                    iteration += 1

        #concatenate series into dataframe
        results = pd.concat(results, axis = 1).T

        analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank','Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)']]
        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank','Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)'], axis = 1)
        results = pd.concat([analyzed_ptms, results])
        #add to ptm_info dataframe
        self.ptm_info = self.ptm_info.join(results) 

        self.ptm_info = self.ptm_info.drop_duplicates()
        print('All data analyzed: saving full ptm_info results')
        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')

            
    def explode_PTMinfo(self, explode_cols = ['Genes', 'Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon Rank', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to N-terminal Splice Boundary (NC)']):
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
        seq = self.transcripts.loc[transcript, 'Amino Acid Sequence']
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
        
    def getFlankingSeq(self, pos, transcript, flank_size = 4):
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
        
    def getAllFlankingSeqs(self, flank_size = 4):
        """
        Runs getAllFlankingSeqs() for all ptms recorded in self.ptm_info. Adds 'Flanking Sequence' column to self.ptm_info after running.
        """
        flanks = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Get flanking sequences'):
            pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
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
        
    def mapPTMsToAlternative(self):
        self.alternative_ptms = alternative_mapping.mapBetweenTranscripts_all(self, results = self.alternative_ptms)
    
    def calculate_PTMconservation(self):
        if self.alternative_ptms is None:
            raise AttributeError('No alternative_ptms attribute. Must first map ptms to alternative transcripts with mapPTMsToAlternative()')
        else:
            #conserved transcripts
            conserved_transcripts = self.alternative_ptms[self.alternative_ptms['Mapping Result'] == 'Success'].groupby('Canonical PTM')['Alternative Transcript'].apply(list)
            num_conserved_transcripts = conserved_transcripts.apply(len)
            conserved_transcripts = conserved_transcripts.apply(','.join)
            
            #lost transcripts
            lost_transcripts = self.alternative_ptms[self.alternative_ptms['Mapping Result'] != 'Success'].groupby('Canonical PTM')['Alternative Transcript'].apply(list)
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
                    conservation_score.append(np.nan)
                #check if any lost transcripts: if not replace NaN with 0 when calculating
                elif num_lost != num_lost:
                    conservation_score.append(1)
                else:
                    conservation_score.append(num_conserved/(num_conserved+num_lost))
            self.ptm_info['PTM Conservation Score'] = conservation_score
        
    def addSpliceEventsToAlternative(self, splice_events_df):
        splice_events_df = [['Exon ID (Canonical)', 'Alternative Transcript', 'Event Type']]
        splice_events_df = splice_events_df.rename({'Exon ID (Canonical)': 'Canonical Exon'}, axis = 1)
        self.alternative_ptms = self.alternative_ptms.merge(splice_events_df, on = ['Canonical Exon', 'Alternative Transcript'], how = 'left')
    
    def annotateAlternativePTMs(self):
        mapper.alternative_ptms['Canonical PTM'] = mapper.alternative_ptms['Protein'] + '_' + mapper.alternative_ptms['Residue'] + mapper.alternative_ptms['Canonical Protein Location (AA)'].astype(str)
        if 'TRIFID Score' in self.transcripts.columns:
            self.alternative_ptms = self.alternative_ptms(self.transcripts['TRIFID Score'], right_index = True, left_on = 'Alternative Transcript', how = 'left')
            
        print('Calculating the rate of PTM conservation for each PTM')
        self.calculate_PTMconservation()
        
        print('Getting flanking sequences around PTMs in alternative isoforms')
        flank = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Location (AA)'], self.alternative_ptms['Alternative Transcript']):
            flank.append(self.getFlankingSeq(pos, transcript_id, flank_size = 10))
        self.alternative_ptms['Flanking Sequence'] = flank
        
        print('Getting tryptic fragments that include each PTM in alternative isoforms')
        tryptic = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Location (AA)'], self.alternative_ptms['Alternative Transcript']):
            tryptic.append(self.getTrypticFragment(pos, transcript_id))
        self.alternative_ptms['Tryptic Fragment'] = tryptic
        
        if os.path.exists(config.processed_data_dir + 'splice_events.csv'):
            sevents = pd.read_csv(config.processed_data_dir + 'splice_events.csv')
            self.addSpliceEventsToAlternative(self, sevents)
            
            
        
            
        
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
        else:
            raise ValueError(f'{exon_sequences_fname} not found in the source data directory. Make sure file is located in the source data directory and matches the directory and file name')
    
    if 'Transcript Sequence' not in mapper.transcripts.columns or restart_all:
        print('Extracting transcript sequence information')
        #load coding sequences
        if os.path.exists(config.source_data_dir+coding_sequences_fname):
            coding_sequences = utility.processEnsemblFasta(config.source_data_dir+'coding_sequences.fasta.gz', id_col = 'Transcript stable ID', seq_col = 'Coding Sequence')
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
            
    if 'Exon AA Seq (Full Codon)' not in mapper.exons.columns or restart_all:
        print('Getting exon-specific amino acid sequence\n')
        mapper.exons = processing.getAllExonSequences(mapper.exons, mapper.transcripts)
        
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
        mapper.mapPTMsToAlternative()
        print('saving\n')
        
    if not os.path.exists(config.processed_data_dir + 'splice_events.csv'):
        print('Identify splice events that result in alternative isoforms')
        splice_events_df = get_splice_events.identifySpliceEvents_All(mapper.exons, mapper.proteins, mapper.transcripts, mapper.genes)
        priny('saving\n')
        splice_events_df.to_csv(config.processed_data_dir + 'splice_events.csv')
        
    if 'Tryptic Fragment' not in mapper.alternative_ptms.columns:
        print('Annotate PTM sites on alternative isoforms')
        mapper.annotateAlternativePTMs()
        print('saving\n')
        mapper.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv')
        
    return mapper
    
