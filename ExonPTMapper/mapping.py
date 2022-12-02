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
from ExonPTMapper import config


class PTM_mapper:
    def __init__(self, exons, transcripts, ptm_info = None):
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
        self.exons = exons
        self.transcripts = transcripts
        #self.transcripts.index = transcripts['Transcript stable ID']
        
        #identify available transcripts for PTM analysis (i.e. those that match in gencode and PS)
        
        
        #if ptms not provided, identify
        if ptm_info is not None:
            self.ptm_info = ptm_info

        
    
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
        if type(ptms) == int:
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
                trans = str(trans)
                info = self.find_ptms(trans)
                if type(info) is str:		   
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
            grouped_mods = self.ptm_info.groupby(level = 0)['Modification'].apply(set).apply(','.join)
            self.ptm_info['Modifications'] = grouped_mods
            self.ptm_info = self.ptm_info.drop(['Gene','Transcript','Modification'], axis = 1)
            self.ptm_info = self.ptm_info.drop_duplicates()
        else:
            print('Multiprocessing not active yet. Please use PROCESSES = 1')
        
    def mapPTM(self, ptm):
        """
        Given a ptm (in the form of 'UniprotID_ResiduePosition'), find where the start of the codon producing the residue is found in the exon, 
        transcript, and gene.
        
        Parameters
        ----------
        ptm: strings
            ptm to map to genome. Example: 'P00533_Y1042'
        
        Returns
        -------
        PTM_start: list or string (depending on the number of transcripts protein is associated with)
            location in transcript of codon associated with PTM residue
        exon_id: list or string (depending on the number of transcripts the protein is associated with)
            exon ensemble id for the exon that the PTM codon is found
        exon_codon_start: list or string (depending on the number of transcripts the protein is associated with)
            location in exon of codon associated with PTM residue
        gene_codon_start: list or string (depending on the number of transcripts/genes the protein is associated with
            location in gene of codon associated with PTM residue
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
            exon_aa_start = []
            for t in transcript_ids:
                if self.transcripts.loc[t, 'Relative CDS Start (bp)'] == 'error:no match found':
                    exon_id.append('CDS fail')
                    exon_rank.append('CDS fail')
                    PTM_start.append('CDS fail')
                    exon_codon_start.append('CDS fail')
                    gene_codon_start.append('CDS fail')
                elif self.exons[self.exons['Transcript stable ID'] == t].shape[0] == 0:
                    exon_id.append('Exons Not Found')
                    exon_rank.append('Exons Not Found')
                    PTM_start.append('Exons Not Found')
                    exon_codon_start.append('Exons Not Found')
                    gene_codon_start.append('Exons Not Found')
                else:
                    CDS_start = int(self.transcripts.loc[t, 'Relative CDS Start (bp)'])
            
                    #calculate location of PTM in transcript (aa pos * 3 -3 + start of coding sequence)
                    start = CDS_start + (position*3-3)
                    PTM_start.append(str(start))

                    #find which exon
                    exon_info = self.exons.loc[self.exons['Transcript stable ID'] == t]
                    exon_row = (start < exon_info['Exon End (Transcript)']) & (start >= exon_info['Exon Start (Transcript)'])
                    exon_of_interest = exon_info[exon_row].squeeze()
                    exon_id.append(exon_of_interest['Exon stable ID'])
                    exon_rank.append(str(exon_of_interest['Exon rank in transcript']))
            
                    #find location in exon and gene
                    ec_start = start - int(exon_of_interest['Exon Start (Transcript)'])+1
                    exon_codon_start.append(str(ec_start))
                    if exon_of_interest['Exon Start (Gene)'] != 'no match' and exon_of_interest['Exon Start (Gene)'] != 'gene not found':
                        if exon_of_interest['Strand'] == 1:
                            gene_codon_start.append(str(ec_start - int(exon_of_interest['Exon Start (Transcript)']) + int(exon_of_interest['Exon Start (Gene)'])))
                        else:
                            gene_codon_start.append(str(int(exon_of_interest['Exon End (Gene)']) - ec_start))
                    else:
                        gene_codon_start.append('not available')
                        
                    #find aa position in exon
                    if exon_of_interest['Exon Start (Protein)'] == 'Partial start codon':
                        exon_aa_start.append('Translation error')
                    else:
                        exon_aa_start.append(str(position - float(exon_of_interest['Exon Start (Protein)'])))
                
            #convert lists to strings
            PTM_start = ','.join(PTM_start)   
            exon_id = ','.join(exon_id) 
            exon_rank = ','.join(exon_rank)
            exon_codon_start = ','.join(exon_codon_start)
            gene_codon_start = ','.join(gene_codon_start)
            exon_aa_start = ','.join(set(exon_aa_start))
        else:
            if self.transcripts.loc[transcript_ids[0], 'Relative CDS Start (bp)'] == 'error:no match found':
                exon_id = 'CDS fail'
                exon_rank = 'CDS fail'
                PTM_start ='CDS fail'
                exon_codon_start = 'CDS fail'
                gene_codon_start = 'CDS fail'
                exon_aa_start = 'CDS fail'
            else:
                CDS_start = int(self.transcripts.loc[transcript_ids[0], 'Relative CDS Start (bp)'])
            
                #calculate location of PTM in transcript (aa pos * 3 -3 + start of coding sequence)
                PTM_start = CDS_start + (position*3-3)
            

                #find which exon
                exon_info = self.exons.loc[self.exons['Transcript stable ID'] == transcript_ids[0]]
                exon_row = (PTM_start < exon_info['Exon End (Transcript)']) & (PTM_start >= exon_info['Exon Start (Transcript)'])
                exon_of_interest = exon_info[exon_row].squeeze()
                exon_id = exon_of_interest['Exon stable ID']
                exon_rank = exon_of_interest['Exon rank in transcript']
        
                #find location in exon and gene
                exon_codon_start = PTM_start - int(exon_of_interest['Exon Start (Transcript)'])+1
                if exon_of_interest['Exon Start (Gene)'] != 'no match' and exon_of_interest['Exon Start (Gene)'] != 'gene not found':
                    #check if gene is on forward or reverse strand (impacts meaning of coordinates)
                    if exon_of_interest['Strand'] == 1:
                        gene_codon_start = exon_codon_start + int(exon_of_interest['Exon Start (Gene)'])
                    else:
                        gene_codon_start = int(exon_of_interest['Exon End (Gene)']) - exon_codon_start
                else:
                    gene_codon_start = 'not available'
                if exon_of_interest['Exon Start (Protein)'] == 'Partial start codon':
                    exon_aa_start = 'Translation error'
                else:
                    exon_aa_start = str(position - float(exon_of_interest['Exon Start (Protein)']))

        
        return pd.Series(data = [gene_codon_start, PTM_start, exon_codon_start, exon_aa_start, exon_id, exon_rank],
                        index = ['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank'],
                        name = ptm)
                        
        
    def mapPTMs_all(self, save_iter = 5000, restart = False, PROCESSES = 1):
        """
        For all ptms in ptm_info, map to genome and save in ptm_positions 
        """
        #process ptms to operate on each unique PTM instance separately
        
        #exploded_ptms = self.explode_PTMinfo()
        
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
                analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank']]
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
                        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank'], axis = 1)
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
                            self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank'], axis = 1)
                        self.ptm_info = self.ptm_info.join(tmp_results)
                        
                        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
                    iteration += 1

        #concatenate series into dataframe
        results = pd.concat(results, axis = 1).T

        analyzed_ptms = self.ptm_info.loc[~self.ptm_info['Exon stable ID'].isna(),['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank']]
        self.ptm_info = self.ptm_info.drop(['Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon Location (AA)', 'Exon stable ID', 'Exon Rank'], axis = 1)
        results = pd.concat([analyzed_ptms, results])
        #add to ptm_info dataframe
        self.ptm_info = self.ptm_info.join(results) 

        
        print('All data analyzed: saving full ptm_info results')
        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')

            
    def explode_PTMinfo(self, explode_cols = ['Genes', 'Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon Rank']):
        exploded_ptms = self.ptm_info.copy()
        #check columns that exist
        explode_cols = [col for col in explode_cols in col in exploded_ptms.columns.values]
        #split different entries
        for col in explode_cols:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(','))
      
        exploded_ptms = exploded_ptms.explode(explode_cols)
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

        
    def getTrypticFragment(self, ptm):
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
        pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
        transcript = self.ptm_info.loc[ptm, 'Transcripts']
        #if multiple transcripts associated with protein, only use first transcript (should be same seq)
        if ',' in transcript:
            transcript = transcript.split(',')[0]
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
            fragment.append(self.getTrypticFragment(ptm))
        self.ptm_info['Tryptic Fragment'] = fragment
        
    def getFlankingSeq(self, ptm, flank_size = 4):
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
        pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
        transcript = self.ptm_info.loc[ptm, 'Transcripts']
        #if multiple transcripts associated with protein, only use first transcript (should be same seq)
        if ',' in transcript:
            transcript = transcript.split(',')[0]
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
            flanks.append(self.getFlankingSeq(ptm, flank_size))
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
        
        
    def distance_to_boundary(self, ptm):
        """
        Find the number of residues between the indicated ptm and the closest splice boundary
        
        Parameters
        ----------
        ptm: str
            ptm to get flanking sequence, indicated by 'SubstrateID_SiteNum'
        aa_cuts: list of floats
            location of splice boundaries (based on amino acids)
        
        Returns
        -------
        min_distance: float
            indicates the number of residues from the closest splice boundary
        """
        #get ptm position (residues)
        exon_pos = float(self.ptm_info.loc[ptm,'Exon Location (AA)'])
        exon_id = self.ptm_info.loc[ptm, 'Exon stable ID'].split(',')
        if any([True if (i != 'Exons Not Found' and i != 'No coding seq') else False for i in exon_id]):
            
            for i in range(len(exon_id)):
                if exon_id[i] != 'Exons Not Found' and exon_id[i] != 'No coding seq':
                    index = i
                    break
                
            exon_id = exon_id[index]
            try:
                exon_start = float(self.exons.loc[self.exons['Exon stable ID'] == exon_id, 'Exon Start (Protein)'].values[0])
                exon_end = float(self.exons.loc[self.exons['Exon stable ID'] == exon_id, 'Exon End (Protein)'].values[0])
            except:
                return np.nan, np.nan
            n_term = exon_pos - exon_start
            c_term = exon_end - exon_pos
        
            return n_term, c_term
        else:
            return np.nan, np.nan
        
        
    #def distance_to_boundary(self, ptm):
    #    """
    #    Find the number of residues between the indicated ptm and the closest splice boundary
    #    
    #    Parameters
    #    ----------
    #    ptm: str
    #        ptm to get flanking sequence, indicated by 'SubstrateID_SiteNum'
    #    aa_cuts: list of floats
    #        location of splice boundaries (based on amino acids)
    #    
    #    Returns
    #    -------
    #    min_distance: float
    #        indicates the number of residues from the closest splice boundary
    #    """
    #    #get ptm position (residues)
    #    pos = int(self.ptm_info.loc[ptm,'PTM Location (AA)'])
    #    ranks = self.ptm_info.loc[ptm, 'Exon Rank'].split(',')
    #    if any([True if rank != 'Exons Not Found' else False for rank in ranks]):
    #        
    #        for i in range(len(ranks)):
    #            if ranks[i] != 'Exons Not Found':
    #                index = i
    #                break
    #            
    #        rank = int(ranks[index])
    #    
    #        #get exon cuts and convert to amino acids
    #        transcript_id = self.ptm_info.loc[ptm, 'Transcripts'].split(',')[index]
    #        #add 0 to list to indicate start of transcript
    #        nc_cuts = [0]+self.transcripts.loc[transcript_id, 'Exon cuts'].split(',')
    #        
    #        #get n and c-terminal cuts, convert to residue location
    #        nc_cuts = [nc_cuts[rank-1],nc_cuts[rank]]
    #        cds_start = self.transcripts.loc[transcript_id, 'Relative CDS Start (bp)']
    #        nc_pos = Prot_to_RNA(pos, cds_start)
    #        #aa_cuts = [RNA_to_Prot(cut, cds_start) for cut in nc_cuts]
    #        #calculate distance between PTM and closest N-terminal splice boundary
    #        n_distance = nc_pos - int(nc_cuts[0])
    #        #calculate distance between PTM and closest C-terminal splice boundary
    #        c_distance = int(nc_cuts[1]) - nc_pos
    #    
    #        return n_distance, c_distance
    #    else:
    #        return np.nan, np.nan
        
    def boundary_analysis(self):
        """
        Find how close each ptm is to the splice boundaries
        
        """
        c_boundary = []
        n_boundary = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Getting distance to boundary'):
            results = self.distance_to_boundary(ptm)
            n_boundary.append(results[0])
            c_boundary.append(results[1])
            
        self.ptm_info['Distance to N-term (AA)'] = n_boundary
        self.ptm_info['Distance to C-term (AA)'] = c_boundary
        
    def PTMs_inAlternative(self):
        print('Not ready')
        
        
    def savePTMs(self):
        self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        self.ptm_positions.to_csv(config.processed_data_dir + 'ptm_positions.csv')
        
def load_PTMmapper():
    exons = pd.read_csv(config.processed_data_dir + 'exons.csv')
    transcripts = pd.read_csv(config.processed_data_dir + 'transcripts.csv', index_col = 0)
    if os.path.exists(config.processed_data_dir + 'ptm_info.csv'):
        ptm_info = pd.read_csv(config.processed_data_dir + 'ptm_info.csv',index_col = 0)
        mapper = PTM_mapper(exons, transcripts, ptm_info)
    else:
        mapper = PTM_mapper(exons, transcripts)
    #ptm_positions = pd.read_csv(config.processed_data_dir + 'ptm_positions.csv', index_col = 0)
    #ptms_all = pd.concat([ptm_info, ptm_positions], axis = 1)
    #ptms_all = ptms_all.loc[:,~ptms_all.columns.duplicated()]
    return mapper
	
    
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

          

    
