import pandas as pd
import numpy as np

import multiprocessing
import contextlib
import io

#Biopython
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from cogent3 import make_aligned_seqs

#file processing
import os
import re
import pickle
import logging

#Other packages
from tqdm import tqdm
import pyliftover

#ExonPTMapper packages
from ExonPTMapper import config, processing, utility, get_splice_events

#PTM-POSE package
#from PTM_POSE import project as pose_project

#initialize logger
logger = logging.getLogger('Mapping')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(config.processed_data_dir + 'ExonPTMapper.log')
log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(log_format)
logger.addHandler(handler)


class PTM_mapper:
    def __init__(self, from_pickle = False):
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
        self.load_PTMmapper(from_pickle = from_pickle)

    def find_ptms_one_protein(self, uniprot_id, collapse = True, phosphositeplus_data = None):
        """
        Given a uniprot ID, find all PTMs present in the protein and save to dataframe
        
        Parameters
        ----------
        unip_id: strings
            Ensemble transcript for the protein of interest
            
        Returns
        -------
        ptm_df: pandas dataframe
            Dataframe containing gene id, transcript id, protein id, residue modified, location of residue and modification type. Each row
                corresponds to a unique ptm
        """
        gene_id = self.proteins.loc[uniprot_id, 'Gene stable ID']
        transcript_id = self.proteins.loc[uniprot_id, 'Associated Matched Transcripts']
        iso_type = self.proteins.loc[uniprot_id, 'UniProt Isoform Type']
        #if pscout has any matched transcripts, get PTMs from proteomescout
        if len(set(transcript_id.split(';')).intersection(config.pscout_matched_transcripts)) > 0:
            ptms_pscout = config.ps_api.get_PTMs(uniprot_id.replace('-','.'))
            #if labeled as the canonical isoform, check the base uniprot id as well if nothing found initiall
            if isinstance(ptms_pscout, int) and iso_type == 'Canonical':
                ptms_pscout = config.ps_api.get_PTMs(uniprot_id.split('-')[0])
        else:
            ptms_pscout = -1

        #if phosphosite plus file provided and has matching transcripts, get PTMs from phosphositeplus
        if phosphositeplus_data is not None and len(set(transcript_id.split(';')).intersection(config.psp_matched_transcripts)) > 0:
            ptms_psp = utility.get_PTMs_PhosphoSitePlus(uniprot_id, phosphositeplus_data, isoform_type = iso_type)
        else:
            ptms_psp = -1

        #extract ptm position, if any found
        if isinstance(ptms_pscout, int) and isinstance(ptms_psp, int):
            return pd.DataFrame()
        elif isinstance(ptms_psp, int):
            ptm_df = pd.DataFrame(ptms_pscout, columns = ['PTM Location (AA)', 'Residue', 'Modification'])
            ptm_df['Sources'] = 'ProteomeScout'
        elif isinstance(ptms_pscout, int):
            ptm_df = pd.DataFrame(ptms_psp, columns = ['PTM Location (AA)', 'Residue', 'Modification'])
            ptm_df['Sources'] = 'PhosphoSitePlus'
        else: 
            ptm_pscout_df = pd.DataFrame(ptms_pscout, columns = ['PTM Location (AA)', 'Residue', 'Modification'])
            ptm_pscout_df['Sources'] = 'ProteomeScout'
            ptm_psp_df = pd.DataFrame(ptms_psp, columns = ['PTM Location (AA)', 'Residue', 'Modification'])
            ptm_psp_df['Sources'] = 'PhosphoSitePlus'
            ptm_df = pd.concat([ptm_pscout_df, ptm_psp_df])

        #add broader modification class to PTMs
        ptm_df = ptm_df.merge(config.modification_conversion[['Modification','Modification Class']], on = 'Modification', how = 'left')

        #if any nan values in modification class, replace with same as modification
        ptm_df['Modification Class'] = ptm_df['Modification Class'].fillna(ptm_df['Modification'])

        #aggregate ptms to avoid duplicates from phosphositeplus and proteomescout or duplicate sites (some sites can be modified in multiple ways)
        if collapse:
            ptm_df = ptm_df.groupby(['PTM Location (AA)', 'Residue'], as_index = False).agg(utility.join_unique_entries)
            ptm_df.index = uniprot_id + '_' + ptm_df['Residue'] + ptm_df['PTM Location (AA)'].astype(str)
        else:
            ptm_df = ptm_df.groupby(['PTM Location (AA)', 'Residue', 'Modification', 'Modification Class'], as_index = False).agg(utility.join_unique_entries)
            ptm_df['PTM'] = uniprot_id + '_' + ptm_df['Residue'] + ptm_df['PTM Location (AA)'].astype(str)

        #ptm_df.columns = ['PTM Location (AA)', 'Residue', 'Modification', 'Source']
        ptm_df.insert(0, 'Isoform Type', iso_type)
        ptm_df.insert(0, 'Protein', uniprot_id)
        ptm_df.insert(0, 'Transcripts', transcript_id)
        ptm_df.insert(0, 'Genes', gene_id)
        #get gene name from gene_id
        gene_id = gene_id.split(';')[0]
        ptm_df.insert(0, 'Gene name', self.genes.loc[gene_id, 'Gene name'])
        #ptm_df.index = ptm_df['Protein']+'_'+ptm_df['Residue']+ptm_df['PTM Location (AA)']
    
        return ptm_df
               
    def find_ptms_list(self, uniprot_ids, collapse = True, phosphositeplus_data = None):
        """
        Given a list of uniprot IDs, find all PTMs present in the proteins and save to dataframe
        
        Parameters
        ----------
        unip_id: list
            Ensemble transcript for the protein of interest
            
        Returns
        -------
        ptm_df: pandas dataframe
            Dataframe containing gene id, transcript id, protein id, residue modified, location of residue and modification type. Each row
                corresponds to a unique ptm
        """
        num_ptms = {}
        df_list = []
        for prot in tqdm(uniprot_ids, desc = 'Finding PTMs for all proteins with matched transcripts'):
            #check to make sure transcript has appropriate information
            info = self.find_ptms_one_protein(prot, phosphositeplus_data = phosphositeplus_data, collapse = collapse)

            if info.empty:
                num_ptms[prot] = 0
            else:
                df_list.append(info)	
                if collapse:
                    num_ptms[prot] = info.shape[0]
                else:
                    num_ptms[prot] = info['PTM'].nunique()
        ptm_df = pd.concat(df_list).dropna(axis = 1, how = 'all')
        ptm_df['PTM Location (AA)'] = ptm_df['PTM Location (AA)'].astype(int)
        return ptm_df, num_ptms
        
    def find_ptms_all(self, phosphositeplus_file = None, collapse = True, PROCESSES = 1):
        """
        Run find_ptms() for all proteins with available matched transcripts, save in ptm_info dataframe.

        Parameters
        ----------
        collapse: bool
            Indicates whether rows should be collapsed on unique modifications
        PROCESSES: int
            Number of processes to run simultaneously, currently only 1 is allowed
        """
        #record start of finding ptms in logger
        if collapse:
            logger.info('Getting PTMs associated with proteins with matching transcripts in Ensembl. Each row in ptm info dataframe will be specific to a modified residue.')
        else:
            logger.info('Getting PTMs associated with proteins with matching transcripts in Ensembl. Each row in ptm info dataframe will be specific to a modification, so some rows will be associated with the same residue.')
        #load phosphositeplus data if available
        if phosphositeplus_file is not None:
            phosphositeplus_data = pd.read_csv(phosphositeplus_file, index_col = 0)
        else:
            phosphositeplus_data = None

        if PROCESSES == 1:
            #remove proteins without matched transcripts
            trim_proteins = self.proteins.dropna(subset = 'Associated Matched Transcripts').copy()
            ptm_info, num_ptms = self.find_ptms_list(trim_proteins.index.values, collapse = collapse, phosphositeplus_data = phosphositeplus_data)
            

            #combine all protein information into one dataframe
            self.ptm_info = ptm_info

            #add number of ptms information to protein dataframe
            num_ptms = pd.Series(num_ptms, name = 'Number of PTMs')
            self.proteins['Number of PTMs'] = num_ptms


        else:
            #check num_cpus available, if greater than number of cores - 1 (to avoid freezing machine), then set to PROCESSES to 1 less than total number of cores
            num_cores = multiprocessing.cpu_count()
            if PROCESSES > num_cores - 1:
                PROCESSES = num_cores - 1
            
            #grab protein information with at least one matching transcript in ensembl
            trim_proteins = self.proteins.dropna(subset = 'Associated Matched Transcripts').copy()
            #split dataframe into chunks equal to PROCESSES
            protein_data_split = np.array_split(trim_proteins.index.values, PROCESSES)
            pool = multiprocessing.Pool(PROCESSES)
            #run with multiprocessing
            results = pool.starmap(self.find_ptms_list, [(protein_data_split[i], collapse, phosphositeplus_data) for i in range(PROCESSES)])

            #extract info from run
            self.ptm_info = pd.concat([res[0] for res in results])
            num_ptms_list = [res[1] for res in results]
            num_ptms = {}
            for item in num_ptms_list:
                num_ptms.update(item)
            
            #add number of ptms information to protein dataframe
            num_ptms = pd.Series(num_ptms, name = 'Number of PTMs')
            self.proteins['Number of PTMs'] = num_ptms


    
        
    def mapPTMs_all(self, restart = False, PROCESSES = 1):
        """
        For all ptms in ptm_info, map to their respective exon and their location in the genome. Will also create a genomic coordinate specific dataframe called ptm_coordinates which will be used for mapping modifications onto alternative transcripts 

        Parameters
        ----------
        save_iter: int
            Number of ptms to iterate through before saving data to tmp csv (not currently in use)
        restart: bool
            indicates whether to fully restart mapping process (not currently in use)
        PROCESSES: int
            indicates how many processes to run simultaneously (not currently in use)

        Returns
        -------
        None, but updates ptm_info attribute with location of ptm in exon, gene, and transcript, and creates new attribute called ptm_coordinates with unique info specific to genomic locations of ptms (for use with mapping to alternative transcripts)
        """
        #create copy of existing ptm_info dataframe
        ptm_info = self.ptm_info.copy()
        ptm_info = ptm_info.rename({'Transcript':'Transcripts'}, axis = 1)
        #separate ptm_info dataframe by unique transcripts
        ptm_info['Transcripts'] = ptm_info['Transcripts'].apply(lambda x: x.split(';'))
        ptm_info = ptm_info.explode('Transcripts')

        print('Getting location of PTMs in exon, transcript, and gene')
        logger.info('Getting location of PTMs in exon, transcript, and gene')

        #extract transcript level info required for mapping process (coding start and location of splice boundaries)
        transcript_data = self.transcripts[['Relative CDS Start (bp)', 'Exon cuts']].copy()
        #remove transcripts without necessary data (nan, or with error string)
        transcript_data = transcript_data.dropna(subset = ['Exon cuts', 'Relative CDS Start (bp)'])

        #convert exon cuts into list containing integers rather than a single string
        transcript_data['Exon cuts'] = transcript_data['Exon cuts'].apply(lambda cut: np.array([int(x) for x in cut.split(',')]))
        transcript_data = transcript_data.dropna(subset = ['Exon cuts', 'Relative CDS Start (bp)'])

        #add transcript data to ptm information
        ptm_info = ptm_info.merge(transcript_data, left_on = 'Transcripts', right_index = True, how = 'left')
        ptm_info = ptm_info.dropna(subset = 'Exon cuts')

        #get transcript location of PTMs
        ptm_info['Transcript Location (NC)'] = ((ptm_info['PTM Location (AA)'].astype(int)-1)*3 + ptm_info['Relative CDS Start (bp)'].astype(int))
        
        
        #get rank of exon in transcript, based on transcript location and exon cuts. To do so, find the first exon cut which is greater than transcript location.
        min_exon_rank = self.exons.groupby('Transcript stable ID')['Exon rank in transcript'].min()
        exon_rank = []
        n_dist_list = []
        c_dist_list = []
        min_dist_list = []
        ragged_list = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Identify PTM-containing exons'):
            #get distance of PTM from splice boundaries by subtracting PTM location in transcript by splice boundaries
            normed_cuts = row['Transcript Location (NC)'] - row['Exon cuts']
            #find the first negative number in normed_cuts (larger than transcript loc), which will indicate the correct exon rank
            for c in range(len(normed_cuts)):
                if normed_cuts[c] <  0:
                    #add 1, as currently in pythonic coordinates (if missing first exon ranks, add additional)
                    exon_rank.append(c+min_exon_rank[row['Transcripts']])

                    #record distance n-terminal boundary/start of exon (if exon rank is 1, this will be the same as the transcript location, else use the normed cuts)
                    if c == 0:
                        n_distance = row['Transcript Location (NC)']
                    else:
                        n_distance = normed_cuts[c-1]
                    
                    #record distance to c-terminal boundary/end of exon
                    c_distance = row['Exon cuts'][c] - (row['Transcript Location (NC)'] + 3)

                    #record the minimum distance to boundary
                    min_distance = min([n_distance, c_distance])

                    #assess if ptm is ragged (coded for by two exons, found at splice boundary)
                    ragged = min_distance < 0

                    #save data to lists
                    n_dist_list.append(n_distance)
                    c_dist_list.append(c_distance)
                    min_dist_list.append(min_distance)
                    ragged_list.append(ragged)
                    break
        
        #save data to columns in dataframe
        ptm_info['Exon rank in transcript'] = exon_rank
        ptm_info['Distance to N-terminal Splice Boundary (NC)'] = n_dist_list
        ptm_info['Distance to C-terminal Splice Boundary (NC)'] = c_dist_list
        ptm_info['Distance to Closest Boundary (NC)'] = min_dist_list
        ptm_info['Ragged'] = ragged_list

        #remove exon cuts column, no longer needed
        ptm_info = ptm_info.drop('Exon cuts', axis = 1)
        
        #add exon level information required for rest of mapping process (mapping to genomic location)
        exon_info = self.exons[['Transcript stable ID', 'Exon stable ID', 'Exon rank in transcript', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Protein)', 'Exon End (Protein)', 'Warnings (AA Seq)']].copy()
        ptm_info = ptm_info.merge(exon_info, left_on = ['Transcripts', 'Exon rank in transcript'], right_on = ['Transcript stable ID', 'Exon rank in transcript'], how = 'left')

        
        #add gene info to ptm dataframe (which strand and chromosome ptm is located)
        gene_info = ptm_info.apply(lambda x: self.genes.loc[x['Genes'].split(';')[0], ['Chromosome/scaffold name', 'Strand']], axis = 1)
        ptm_info = pd.concat([ptm_info, gene_info], axis = 1)
        #ptm_info = ptm_info.merge(self.genes[['Chromosome/scaffold name', 'Strand']], left_on = 'Genes', right_index = True, how = 'left')

        #get genomic locatio of ptms and coordinates
        gene_loc = []
        coordinates = []
        second_exon = []
        ragged_loc_list = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Getting location of PTMs in genome'):
            #check strand of gene, calculate location of ptm in gene based on which strand
            if row['Strand'] == 1:  
                loc = row['Distance to N-terminal Splice Boundary (NC)'] + row['Exon Start (Gene)']
            else:
                loc = row['Exon End (Gene)'] - row['Distance to N-terminal Splice Boundary (NC)']

            #check if able to get location, if so convert to integer
            if loc == loc:
                loc = int(loc)
            gene_loc.append(loc)
                
            #given gene location, get genomic coordinates in the following format (<chromosome>:<start>-<stop>:<strand>)
            if not row['Ragged']:   #genomic coordinates when gene is not found at splice boundary
                coordinates.append(getGenomicCoordinates(row['Chromosome/scaffold name'], loc, row['Strand']))
                second_exon.append(np.nan)
                ragged_loc_list.append(np.nan)
            else:   #genomic coordinates when gene is found at splice boundary (ragged site)
                #identify the other exon contributing to ragged PTM site, get start of this exon 
                next_rank = row['Exon rank in transcript'] + 1
                transcript = self.exons[self.exons['Transcript stable ID'] == row['Transcripts']]
                next_exon = transcript[transcript['Exon rank in transcript'] == next_rank].squeeze()

                #get location of ragged ptm to feed into function
                ragged_loc = int(next_exon['Exon Start (Gene)'] if row['Strand'] == 1 else next_exon['Exon End (Gene)'])
                min_dist = row['Distance to Closest Boundary (NC)']
                #get coordinates of ragged PTM, save exon id of second contributing exon
                coordinates.append(getRaggedCoordinates(row['Chromosome/scaffold name'], loc, ragged_loc, min_dist, row['Strand']))
                ragged_loc_list.append(ragged_loc)
                second_exon.append(next_exon['Exon stable ID'])

        #save data to ptm dataframe
        ptm_info['Gene Location (NC)'] = gene_loc
        ptm_info['Genomic Coordinates'] = coordinates
        ptm_info['Second Contributing Exon'] = second_exon
        ptm_info['Ragged Genomic Location'] = ragged_loc_list
        ptm_info = ptm_info.drop(['Exon Start (Gene)', 'Exon End (Gene)'], axis = 1)
        

        
        #get ptm location in exon in amino acid coordinates
        exon_aa_loc = []
        for i, row in tqdm(ptm_info.iterrows(), total = ptm_info.shape[0], desc = 'Getting residue number within each exon'):
            if row['Warnings (AA Seq)'] == row['Warnings (AA Seq)']:
                exon_aa_loc.append(np.nan)
            else:
                loc = round(row['PTM Location (AA)'] - float(row['Exon Start (Protein)']), 2)
                exon_aa_loc.append(loc)
                
        ptm_info['Exon Location (AA)'] = exon_aa_loc
        
        #add column with ptm name (<UniProtID>_<Residue><Position>)
        ptm_info['PTM'] = ptm_info['Protein'] + '_' + ptm_info['Residue']+ ptm_info['PTM Location (AA)'].astype(str)

        print('Constructing ptm coordinates dataframe')
        logger.info('Constructing ptm coordinates dataframe')
        #save new dataframe which will be trimmed version of ptm info with each row containing a PTM mapped to unique genomic coordinates
        ptm_coordinates = ptm_info[['Genomic Coordinates', 'PTM','Residue', 'Modification', 'Modification Class', 'Chromosome/scaffold name', 'Strand','Gene Location (NC)', 'Ragged', 'Ragged Genomic Location', 'Exon stable ID', 'Gene name']].copy()
        ptm_coordinates = ptm_coordinates.dropna(subset = 'Gene Location (NC)')
        ptm_coordinates = ptm_coordinates.drop_duplicates()
        ptm_coordinates = ptm_coordinates.astype({'Gene Location (NC)': int, 'Strand':int, 'Ragged':bool})
        


        #group modifications for the same ptm in the same row
        grouped = ptm_coordinates.groupby(['Genomic Coordinates', 'Chromosome/scaffold name', 'Residue', 'Strand', 'Gene Location (NC)', 'Ragged'])
        ptm_coordinates = pd.concat([grouped['PTM'].agg(utility.join_unique_entries), grouped['Ragged Genomic Location'].apply(lambda x: np.unique(x)[0]), grouped['Modification'].agg(utility.join_unique_entries), grouped['Modification Class'].agg(utility.join_unique_entries), grouped['Exon stable ID'].agg(utility.join_unique_entries), grouped['Gene name'].agg(utility.join_unique_entries)], axis = 1)
        ptm_coordinates = ptm_coordinates.reset_index()
        ptm_coordinates = ptm_coordinates.rename({'PTM':'Source of PTM', 'Exon stable ID': 'Source Exons', 'Gene Location (NC)':'Gene Location (hg38)'}, axis = 1)

        #annotate with ptm position in canonical isoform
        ptm_coordinates['UniProtKB Accession'] = ptm_coordinates['Source of PTM'].apply(lambda x: x.split(';'))
        ptm_coordinates['Residue'] = ptm_coordinates['UniProtKB Accession'].apply(lambda x: x[0].split('_')[1][0])
        ptm_coordinates['PTM Position in Canonical Isoform'] = ptm_coordinates['UniProtKB Accession'].apply(lambda x: [ptm.split('_')[1][1:] for ptm in x if ptm.split('_')[0] in config.canonical_isoIDs.values()])
        ptm_coordinates['PTM Position in Canonical Isoform'] = ptm_coordinates['PTM Position in Canonical Isoform'].apply(lambda x: ';'.join(x) if len(x) > 0 else np.nan)
        ptm_coordinates['UniProtKB Accession'] = ptm_coordinates['UniProtKB Accession'].apply(lambda x: ';'.join(np.unique([ptm.split('-')[0] for ptm in x])))



        #make genomic coordinates the index of dataframe
        ptm_coordinates = ptm_coordinates.set_index('Genomic Coordinates')

        #reorder column names
        ptm_coordinates = ptm_coordinates[['Gene name', 'UniProtKB Accession', 'Residue', 'PTM Position in Canonical Isoform', 'Modification', 'Modification Class', 'Chromosome/scaffold name', 'Strand', 'Gene Location (hg38)', 'Ragged', 'Ragged Genomic Location', 'Source Exons', 'Source of PTM']]


        #get coordinates in the hg19 version of ensembl using hg38 information using pyliftover
        #hg19_coords = []
        #liftover_object = pyliftover.LiftOver('hg38','hg19')
        #for i, row in tqdm(ptm_coordinates.iterrows(), total = ptm_coordinates.shape[0], desc = 'Converting from hg38 to hg19 coordinates'):
        #    hg19_coords.append(convertToHG19(row['Gene Location (hg38)'], row['Chromosome/scaffold name'], row['Strand'], liftover_object = liftover_object))
        #ptm_coordinates['HG19 Location'] = hg19_coords
        #ptm_coordinates = ptm_coordinates.drop_duplicates()
        
        self.ptm_coordinates = ptm_coordinates.copy()
        
        
        #remove extra data that is not needed for ptm info
        ptm_info = ptm_info.drop(['Warnings (AA Seq)', 'Exon Start (Protein)', 'Exon End (Protein)','Strand', 'Transcript stable ID', 'Relative CDS Start (bp)', 'Chromosome/scaffold name', 'Second Contributing Exon', 'Ragged Genomic Location'], axis = 1)

        #convert to appropriate data types
        ptm_info = ptm_info.astype({'Transcript Location (NC)': int, 'Distance to C-terminal Splice Boundary (NC)': int, 'Distance to N-terminal Splice Boundary (NC)': int, 'Distance to Closest Boundary (NC)': int})


        #collapse into rows for unique ptms, with overlapping info seperated by ;
        ptm_info = ptm_info.astype(str)
        ptm_info = ptm_info.groupby(['PTM','Gene name', 'Genes', 'Protein', 'Isoform Type', 'Residue', 'PTM Location (AA)', 'Modification', 'Modification Class', 'Sources']).agg(';'.join).reset_index()

        #add ptm label to index, remove as column
        ptm_info = ptm_info.set_index('PTM')
        self.ptm_info = ptm_info.copy()
            
    def add_new_coordinate_type(self, to_type = 'hg19'):
        #get coordinates in the hg19 version of ensembl using hg38 information using pyliftover
        new_coords = []
        if to_type == 'hg19':
            liftover_object = pyliftover.LiftOver('hg38',to_type)
            for i, row in tqdm(self.ptm_coordinates.iterrows(), total = self.ptm_coordinates.shape[0], desc = 'Converting from hg38 to hg19 coordinates'):
                new_coords.append(convertToHG19(row['Gene Location (hg38)'], row['Chromosome/scaffold name'], row['Strand'], liftover_object = liftover_object))

        self.ptm_coordinates[f'Gene Location ({to_type})'] = new_coords

            
    def explode_PTMinfo(self, explode_cols = ['Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon rank in transcript', 'Exon Location (AA)', 'Distance to C-terminal Splice Boundary (NC)', 'Distance to N-terminal Splice Boundary (NC)']):
        """
        Expand the ptm_info dataframe into distinct rows for each transcript that the ptm is associated with. Resulting dataframe will have some rows referencing the same ptm, but with data associated with a specific transcript

        Parameters
        ----------
        explode_cols: list
            columns to split into unique entries (items separated by ;), and into distinct rows. Default is all the columns that contain ';' after running mapPTMs_all()

        Returns
        -------
        exploded_ptms: pandas dataframe
            dataframe with each row containing a unique ptm/transcript combination
        """
        exploded_ptms = self.ptm_info.copy()
        #check columns that exist
        explode_cols = [col for col in explode_cols if col in exploded_ptms.columns.values]

        #split different entries
        for col in explode_cols:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(';') if x == x else np.nan)

        #explode columns into distinct rows
        exploded_ptms = exploded_ptms.explode(explode_cols)
        
        #get rid of ptm label index, add as column instead
        exploded_ptms = exploded_ptms.reset_index()
        exploded_ptms = exploded_ptms.rename({'index':'PTM'}, axis = 1)
        return exploded_ptms
        
    def collapse_PTMinfo(self, all_cols = ['Genes', 'Transcripts','Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)','Exon stable ID', 'Exon rank in transcript'], unique_cols = ['Modifications', 'Exon Location (AA)', 'Protein', 'PTM Location (AA)', 'Residue']):
        """
        Collapse the ptm_info dataframe into distinct rows for each ptm (assumes that rows of ptm_info are not unique to a ptm. 

        Parameters
        ----------
        all_cols: list
            columns to collapse into single entry (items separated by ;) such that each row corresponds to a unique PTM. Default is all the columns that contain ';' after running mapPTMs_all()
        unique_cols: list
            columns that correspond to PTM site (not different across entries)

        Returns
        -------
        collapsed_ptms: pandas dataframe
            dataframe with each row containing a unique ptm
        """
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
        #extract amino acid sequence
        seq = self.transcripts.loc[transcript_id, 'Amino Acid Sequence']

        #identify where the tryptic cut would be after the ptm
        c_terminal_cut = re.search('[K|R][^P]|$K|R', seq[pos:])
        if c_terminal_cut is None:  #if search returns none, then the cut is the end of the protein
            c_terminal_cut = len(seq)
        else:
            c_terminal_cut = pos + 1+ c_terminal_cut.span()[0]
            
        #identify where the tryptic cut would be before the ptm
        n_terminal_cut = re.search('^K|R|[^P][K|R]', seq[pos-2::-1])
        if n_terminal_cut is None:  #if search returns none, then the cut is the start of the protein
            n_terminal_cut = 0
        else:
            n_terminal_cut = pos - n_terminal_cut.span()[1]
            
        
        return seq[n_terminal_cut:c_terminal_cut]
    
    def getAllTrypticFragments(self):
        """
        Runs getTrypticFragment() for all ptms recorded in self.ptm_info. Adds 'Tryptic Fragment' column to self.ptm_info after running.

        Returns
        -------
        None, but updates self.ptm_info with 'Tryptic Fragment' column
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
        #force position to int
        pos = int(pos)
        #get protein sequence
        protein_sequence = self.transcripts.loc[transcript, 'Amino Acid Sequence']
        #check if ptm is at the start or end of protein, which will alter how flanking sequence is extracted
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

        Parameters
        ----------
        flank_size: int
            number of residues to return around the ptm
        
        Returns
        -------
        None, but updates self.ptm_info with 'Flanking Sequence' column
        """
        flanks = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Get flanking sequences'):
            #grab necessary info
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
        #grab amino acid sequence associated with transcript
        full_seq = str(self.transcripts.loc[self.transcripts['Transcript stable ID'] == transcript_id, 'Amino Acid Sequence'].values[0])

        #search for flanking sequence in the full sequence
        flank_len = (len(flank_seq)-1)/2
        match = re.search(str(flank_seq), full_seq)
        #check if match was found
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
        Given a ptm, figure out whether the ptm is located in a domain based on info from proteomeScout.
        
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
        protein_type = self.ptm_info.loc[ptm, 'Isoform Type']
        pos = int(self.ptm_info.loc[ptm, 'PTM Location (AA)'])
        
        domains = config.ps_api.get_domains(protein.replace('-','.'), 'uniprot')
        if isinstance(domains, int) and protein_type == 'Canonical':
            domains = config.ps_api.get_domains(protein.split('-')[0], 'uniprot')

        inDomain = False
        domain_type = np.nan
        if not isinstance(domains,int):
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

        Parameters
        ----------
        None.

        Returns
        -------
        None, but adds 'inDomain' and 'Domain Type' columns to self.ptm_info

        """
        inDomain_list = []
        domain_type_list = []
        for ptm in tqdm(self.ptm_info.index, desc = 'Finding location in a domain'):
            with contextlib.redirect_stdout(io.StringIO()): #suppress standard print output
                results = self.findInDomains(ptm)
            inDomain_list.append(results[0])
            domain_type_list.append(results[1])
            
        self.ptm_info['inDomain'] = inDomain_list
        self.ptm_info['Domain Type'] = domain_type_list



    def projectPTM_toExons(self, ptm, trim_exons = None, alternative_only = True):
        """
        Given a ptm and its genomic location, project the ptm onto all exons for which that PTM can be found. In other words, if a ptm is located at the coordinate 100, the ptm would be found in an exon that spans 50-150, but not in an exon that spans 200-300. For each exon for which the PTM is found, will then check to make sure residue is unchanged (such as by a frame shift) and identify it's residue position in the alternative protein isoform. By default, this will only look at exons associated with non-canonical transcripts, but this can be changed by setting alternative_only to False.

        Lastly, there is a rare case in which a residue or PTM is coded for by two exons and exists at the splice boundary (we call it a ragged site). For PTMs existing at a splice boundary, we check for the PTM in both exons, in case only one exon is altered but ragged site is conserved.

        Parameters
        ----------
        ptm: pandas Series
            row from self.ptm_coordinates dataframe
        trim_exons: pandas DataFrame
            trimmed version of self.exons which only includes subset of exons that are assoicated with transcripts with available transcript and amino acid sequence. If not provided, will be generated from self.exons.
        alternative_only: bool
            if True, will only look at exons associated with non-canonical transcripts. If False, will look at all exons associated with transcripts with available transcript and amino acid sequence.
        
        Returns
        -------
        exons: pandas DataFrame
            contains all exons that the ptm is found in, along with the residue position of the ptm in the alternative protein isoform if the residue is unchanged.
        """
            
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
        gene_loc = ptm['Gene Location (hg38)']
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
            
        #check to see if any ptms were found
        if ptm_exons.shape[0] != 0:
            #save info about ptm
            ptm_exons['Source Exons'] = ptm['Source Exons']
            ptm_exons['Source of PTM'] = ptm['Source of PTM']
            ptm_exons['Canonical Residue'] = ptm['Residue'] if ptm['Residue'] == ptm['Residue'] else np.nan
            ptm_exons['Modification'] = ptm['Modification']
            ptm_exons['Modification Class'] = ptm['Modification Class']
            

            #check if ptm is at the boundary (ragged site)
            distance_to_bound = ptm_exons.apply(getDistanceToBoundary, args = (gene_loc, strand), axis = 1)
            ptm_exons['Ragged'] = distance_to_bound <= 1

            #identify location of ptm in new exons (different equations depending on if gene is on the forward or reverse strand, or if site is ragged
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
                #iterate through each ptm, checking if site is ragged, and then calculating its location in isoform
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

                #Save location information
                ptm_exons['Genomic Coordinates'] = coords
                ptm_exons['Frame'] = frame_list
                ptm_exons['Alternative Residue'] = residue_list
                ptm_exons['Alternative Protein Position (AA)'] = aa_pos_list
                ptm_exons['Second Exon'] = second_exon_list

            #save if PTM is ragged
            ptm_exons['Ragged'] = ptm_exons['Ragged'].astype(str)
                
        #if site is ragged and PTM was not found in first contributing exon, check the second contributing exon
        if second_ptm_exons is not None:
            coordinates = []
            frame_list = []
            residue_list = []
            position_list = []
            for i, row in second_ptm_exons.iterrows():
                #get PTM distance to boundary (should be the same for all canonical transcripts). this will indicate how much each exon contributes to ragged site
                dist_to_boundary =  int(self.ptm_info.loc[ptm['Source of PTM'].split(';')[0], 'Distance to Closest Boundary (NC)'].split(';')[0])
                start_second_exon = row['Exon Start (Transcript)']
                codon_start = start_second_exon - (3+dist_to_boundary)
                
                #check frame
                transcript = self.transcripts.loc[row['Transcript stable ID']]
                if transcript['Relative CDS Start (bp)'] != 'No coding sequence' and transcript['Relative CDS Start (bp)'] != 'error:no match found':
                    frame, residue, aa_pos = utility.checkFrame(row, transcript, codon_start, loc_type = 'Transcript', strand = strand, return_residue = True)
                else:
                    frame, residue, aa_pos = np.repeat(np.nan, 3)
                    
                frame_list.append(frame)
                residue_list.append(residue)
                position_list.append(aa_pos)
                ragged_loc = row['Exon Start (Gene)'] if strand == 1 else row['Exon End (Gene)']
                coordinates.append(getRaggedCoordinates(chromosome, gene_loc, ragged_loc, dist_to_boundary, strand))

            #save location information, and then add to ptm_exons dataframe
            second_ptm_exons['Canonical Residue'] = ptm['Residue'] if ptm['Residue'] == ptm['Residue'] else np.nan
            second_ptm_exons['Modification'] = ptm['Modification']
            second_ptm_exons['Modification Class'] = ptm['Modification Class']
            second_ptm_exons['Genomic Coordinates'] = coordinates
            second_ptm_exons['Frame'] = frame_list
            second_ptm_exons['Alternative Residue'] = residue_list
            second_ptm_exons['Alternative Protein Position (AA)'] = position_list
            second_ptm_exons = second_ptm_exons.rename({'Exon stable ID': 'Second Exon'}, axis = 1)
            second_ptm_exons['Source Exons'] = ptm['Source Exons']
            second_ptm_exons['Source of PTM'] = ptm['Source of PTM']
            second_ptm_exons['Ragged'] = True
            second_ptm_exons['Ragged'] = second_ptm_exons['Ragged'].astype(str)
            ptm_exons = pd.concat([ptm_exons, second_ptm_exons])

        if ptm_exons.shape[0] > 0:
            results = ptm_exons.copy()
                    
        return results
        
    def projectPTMs_toIsoformExons(self, alternative_only = True, log_run = True, save_data = True, save_iter = 10000):
        """
        Using ptm_coordinates data and exon coordinates, map PTMs to alternative exons and determine their location in the alternative isoform. This takes a bit of time, so have implemented functions to save data in processed data direcotry as the function runs. If code fails before finishing, can reload partially finished data and continue.

        Parameters
        ----------
        save_iter: int
            number of iterations (PTMs analyzed) to run before saving data to processed data directory

        Returns
        -------
        alternative_ptms: class attribute
            dataframe containing information on PTMs associated with alternative isoforms, including PTMs found in the canonical isoform that could not be projected onto the alternative isoform
        """
        logger.info('Projecting PTMs to alternative exons')
        #get all alternative transcripts (protein coding transcripts not associated with a canonical UniProt isoform) and with available coding info
        available_transcripts = self.transcripts.dropna(subset = ['Transcript Sequence', 'Amino Acid Sequence']).index.values
        if alternative_only:
            alternative_transcripts = config.translator.loc[config.translator['UniProt Isoform Type'] != 'Canonical', 'Transcript stable ID']
            available_transcripts = list(set(available_transcripts).intersection(set(alternative_transcripts)))
        
        #grab exons associated with available transcripts
        trim_exons = self.exons[self.exons['Transcript stable ID'].isin(available_transcripts)]
        trim_exons = trim_exons[['Gene stable ID', 'Exon stable ID', 'Exon Start (Gene)', 'Exon End (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)','Transcript stable ID', 'Exon rank in transcript']].drop_duplicates()


        #add chromosome and strand information from gene dataframe to exon dataframe
        trim_exons = trim_exons.merge(self.genes[['Chromosome/scaffold name', 'Strand']], left_on = 'Gene stable ID', right_index = True)
        
        #check if temporary data from an unfinished run exists. If so, load and start from this point.
        if os.path.exists(config.processed_data_dir + 'temp_alt_ptms.csv'):
            #load data from unfinished run and record which PTMs have already been analyzed
            alt_ptms = pd.read_csv(config.processed_data_dir + 'temp_alt_ptms.csv', dtype = {'Ragged': str})
            alt_ptms = alt_ptms.dropna(subset = 'Source of PTM')
            analyzed_ptms = alt_ptms['Source of PTM'].unique()
            print(f'Found {len(analyzed_ptms)} already analyzed from previous runs')
            if log_run:
                logger.info(f'Found {len(analyzed_ptms)} already analyzed from previous runs, continuing mapping to alternative exons for remaining ptms.')

            #identify which PTMs in the canonical isoform still need to be analyzed
            ptms_to_analyze = self.ptm_coordinates[~self.ptm_coordinates['Source of PTM'].isin(analyzed_ptms)]

            #for each PTM that needs to analyzed, map to alternative exons and add to dataframe
            i = 1
            for index,ptm in tqdm(ptms_to_analyze.iterrows(), total = ptms_to_analyze.shape[0]):
                ptm_exons = self.projectPTM_toExons(ptm, trim_exons = trim_exons)
                #check to make sure PTM was mapped to an alternative exon. If it was, add to alternative dataframe
                if ptm_exons is not None:
                    alt_ptms = pd.concat([alt_ptms, ptm_exons])
                
                #if it is the iteration matching the save interval, save data
                if i % save_iter == 0 and save_data:
                    alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
                i = i + 1
        else:
            if log_run:
                logger.info(f'Did not find data from previous runs. Running on all PTMs')
            alt_ptms = None
            i = 1
            for index,ptm in tqdm(self.ptm_coordinates.iterrows(), total = self.ptm_coordinates.shape[0]):
                ptm_exons = self.projectPTM_toExons(ptm, trim_exons = trim_exons)
                #check to make sure PTM was mapped to an alternative exon. If it was, add to alternative dataframe
                if ptm_exons is not None:
                    if alt_ptms is None:
                        alt_ptms = ptm_exons.copy()
                    else:
                        alt_ptms = pd.concat([alt_ptms, ptm_exons])

                #if it is the iteration matching the save interval, save data  
                if i % save_iter == 0 and save_data:
                    alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
                i = i + 1
        if save_data:
            alt_ptms.to_csv(config.processed_data_dir + 'temp_alt_ptms.csv', index = False)
            print('Saving mapped ptms, final')
        
        #remove residual exon columns that are not needed (were used for mapping purposes, but not useful for final dataframe)
        alt_ptms = alt_ptms.drop(['Exon End (Gene)', 'Exon Start (Gene)', 'Exon Start (Transcript)', 'Exon End (Transcript)', 'Exon rank in transcript'], axis = 1)
              
        #### add ptms that were unsuccessfully mapped to alternative transcripts #######
        #get all transcripts (alternative if restricted) associated with each gene from the proteins dataframe
        if alternative_only:
            prot_to_transcript = self.proteins[self.proteins['UniProt Isoform Type'] == 'Canonical'].copy()
        else:
            prot_to_transcript = self.proteins.copy()



        prot_to_transcript = prot_to_transcript.dropna(subset = 'Variant Transcripts')
        prot_to_transcript['Variant Transcripts'] = prot_to_transcript['Variant Transcripts'].str.split(';')
        prot_to_transcript = prot_to_transcript.explode('Variant Transcripts').reset_index()
        prot_to_transcript = prot_to_transcript[['UniProtKB isoform ID', 'Variant Transcripts']].drop_duplicates()
        #limit to transcripts that were analyzed during the mapping process
        prot_to_transcript = prot_to_transcript[prot_to_transcript['Variant Transcripts'].isin(available_transcripts)]
        prot_to_transcript = prot_to_transcript.rename({'UniProtKB isoform ID':'Protein', 'Variant Transcripts':'Transcript stable ID'}, axis = 1)
        #get canonical PTM information to allow for comparison to alternative info
        #ptms = self.ptm_info.reset_index()[['PTM', 'Protein', 'PTM Location (AA)', 'Exon stable ID', 'Ragged', 'Modification', 'Residue']].drop_duplicates()
        #ptms = ptms.rename({'PTM':'Source of PTM','PTM Location (AA)':'Canonical Protein Location (AA)', 'Residue':'Canonical Residue'}, axis = 1)
        ptms = self.ptm_coordinates[['Source Exons', 'Source of PTM', 'Residue', 'Modification', 'Modification Class']].copy()
        ptms = ptms.rename(columns = {'Residue':'Canonical Residue'})
        ptms['Protein'] = ptms['Source of PTM'].apply(lambda x: [i.split('_')[0] for i in x.split(';')])
        ptms = ptms.explode('Protein')
        #merge canonical PTM info with alternative transcript info
        prot_to_transcript = prot_to_transcript.merge(ptms, on = 'Protein')

        #identify which PTMs were not mapped to alternative transcripts, add information from 'alternative' dataframe to 'alt_ptms' dataframe (bit confusing, should likely change nomenclature)
        potential_ptm_isoform_labels = prot_to_transcript['Transcript stable ID'] + '_' + prot_to_transcript['Source of PTM']
        mapped_ptm_isoform_labels = alt_ptms['Transcript stable ID'] + '_' + alt_ptms['Source of PTM']
        missing = prot_to_transcript[~potential_ptm_isoform_labels.isin(mapped_ptm_isoform_labels)]
        missing = missing.rename({'Transcripts':'Transcript stable ID', 'PTM':'Source of PTM', 'Residue':'Canonical Residue'}, axis = 1)
        missing = missing.drop('Protein', axis = 1)
        #add genes
        
        missing = missing.merge(self.transcripts['Gene stable ID'].reset_index(), on = 'Transcript stable ID', how = 'left')
        #add gene info
        gene_info = missing.apply(lambda x: self.genes.loc[x['Gene stable ID'].split(';')[0], ['Chromosome/scaffold name', 'Strand']], axis = 1)
        missing = pd.concat([missing, gene_info], axis = 1)
        #missing = missing.merge(self.genes[['Chromosome/scaffold name', 'Strand']].reset_index(), on = 'Gene stable ID', how = 'left')
        alt_ptms = pd.concat([alt_ptms, missing])

        #remove duplicates caused during merging process (should have all removed by now, but just in case)
        alt_ptms = alt_ptms.drop_duplicates()

        ####### Now that all PTM information has been mapped to alternative transcripts, annotate with the result of the mapping process #######
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
        
        #save results to mapper object and remove temp file
        if log_run:
            logger.info(f'Finished mapping PTMs to alternative exons. {round(alt_ptms[alt_ptms["Mapping Result"] == "Success"].shape[0]/alt_ptms.shape[0]*100, 2)}% of PTMs were successfully mapped to alternative isoforms.')
        self.alternative_ptms = alt_ptms
        if os.path.isfile(config.processed_data_dir + 'temp_alt_ptms.csv'):
            os.remove(config.processed_data_dir + 'temp_alt_ptms.csv')
        if log_run:
            logger.info(f'Saved ptms found in alternative exons in {config.processed_data_dir}alternative_ptms.csv')
    
    def calculate_PTMconservation(self, transcript_subset = None, isoform_subset = None, save_col = 'PTM Conservation Score', save_transcripts = True, return_score = False, unique_isoforms = True, log_run = True):
        """
        For each PTM, calculate the fraction of transcripts/isoforms for which the PTM was found and is present in the isoform. Have the option to calculate this based on transcripts or on protein isoforms (redundant transcripts with matching protein sequences are removed). Also have the option to look at a subset of transcripts/isoforms.

        Parameters
        ----------
        transcript_subset : list, optional
            List of transcript ids to look at. The default is None, which will use all transcripts in the dataset. If unique_isoform is True and isoform_subset is not None, this will be ignored.
        isoform_subset : list, optional
            List of isoform ids to look at. The default is None, which will use all isoforms in the dataset. Will only be used if unique_isoform is True.
        save_col : str, optional
            Name of column to save the PTM conservation score to. The default is 'PTM Conservation Score'.
        save_transcripts : bool, optional
            Whether to save two lists of the alternative transcripts, one for transcripts containing the PTM and one for transcripts the lack the PTM. If True, four additional columns will be added to ptm_info. The default is True (and has to be in order to work right now)
        return_score : bool, optional
            Whether to return the average PTM conservation score across all PTMs. If true, this will also return the number of isoforms/transcripts assessed. The default is False.
        unique_isoforms : bool, optional
            Whether to look only at unique protein isoforms rather than all transcripts. The default is True.
        
        Returns
        -------
        None or list
            If return_score is True, returns the average PTM conservation score across all PTMs. It will also return the number of unique isoforms assessed (if unique_isoforms is True) or the number of transcripts assessed (if unique isoforms is False).
        
        """
        #extract the conserved and lost transcripts or isoforms, depending on whether unique_isoforms is true
        if unique_isoforms:
            if log_run:
                logger.info('Calculating the fraction of unique alternative isoforms for which PTM is conserved (constitutive rate)')
            #if mapper object does not already have isoform_ptms (PTMs specific to unique isoforms), get them
            if self.isoform_ptms is None:
                self.getIsoformSpecificPTMs()
            
            #check if a specific subset of isoforms or transcripts was provided. If so, restrict isoform ptms to these isoforms
            if isoform_subset is not None:
                alt_ptms = self.isoform_ptms[self.isoform_ptms['Isoform ID'].isin(isoform_subset)].copy()
            elif transcript_subset is not None:
                isoform_subset = self.transcriptToIsoform(transcript_subset)
                alt_ptms = self.isoform_ptms[self.isoform_ptms['Isoform ID'].isin(isoform_subset)].copy()
            else:
                alt_ptms = self.isoform_ptms.copy()
                
            #separate sourc of ptm information
            alt_ptms['Source of PTM'] = alt_ptms['Source of PTM'].apply(lambda x: x.split(';'))
            alt_ptms = alt_ptms.explode('Source of PTM')

            #get alternative isoforms for which the PTM is conserved (i.e. the PTM is present in the isoform and has residue data) or lost (i.e. the PTM is not present in the isoform)    
            conserved_transcripts = alt_ptms[alt_ptms['Mapping Result'] == 'Success'].groupby('Source of PTM')['Isoform ID'].apply(list)
            lost_transcripts = alt_ptms[alt_ptms['Mapping Result'] != 'Success'].groupby('Source of PTM')['Isoform ID'].apply(list)
                
        else:
            #ensure that alternative_ptms has been calculated. If not, raise error
            if self.alternative_ptms is None:
                raise AttributeError('No alternative_ptms attribute. Must first map ptms to alternative transcripts with mapPTMsToAlternative()')
            else:
                if log_run:
                    logger.info('Calculating the fraction of alternative transcripts for which PTM is conserved (constitutive rate)')
                #check if a specific subset of transcripts was provided. If so, restrict alternative_ptms to these transcripts
                if transcript_subset is not None:
                    alt_ptms = self.alternative_ptms[self.alternative_ptms['Alternative Transcript'].isin(transcript_subset)].copy()
                else:
                    alt_ptms = self.alternative_ptms.copy()
            #separate sourc of ptm information
            alt_ptms['Source of PTM'] = alt_ptms['Source of PTM'].apply(lambda x: x.split(';'))
            alt_ptms = alt_ptms.explode('Source of PTM')

            #get alternative isoforms for which the PTM is conserved (i.e. the PTM is present in the isoform and has residue data) or lost (i.e. the PTM is not present in the isoform)    
            conserved_transcripts = alt_ptms[alt_ptms['Mapping Result'] == 'Success'].groupby('Source of PTM')['Alternative Transcript'].apply(list)
            lost_transcripts = alt_ptms[alt_ptms['Mapping Result'] != 'Success'].groupby('Source of PTM')['Alternative Transcript'].apply(list)
        
        #calculate the number of conserved transcripts and collapse into a single string
        num_conserved_transcripts = conserved_transcripts.apply(len)
        conserved_transcripts = conserved_transcripts.apply(','.join)
        
        #calculate the number of lost transcripts and collapse into a single string
        num_lost_transcripts = lost_transcripts.apply(len)
        lost_transcripts = lost_transcripts.apply(','.join)
        
        #save transcript information in ptm_info, if requested
        if save_transcripts:
            self.ptm_info['Number of Conserved Isoforms'] = num_conserved_transcripts
            self.ptm_info['Conserved Isoforms'] = conserved_transcripts
            self.ptm_info['Number of Lost Isoforms'] = num_lost_transcripts
            self.ptm_info['Lost Isoforms'] = lost_transcripts
        
        #for each PTM, calculate the fraction of transcripts/isoforms for which the PTM was found and is present in the isoform
        conservation_score = []
        for ptm in self.ptm_info.index:
            num_conserved = num_conserved_transcripts[ptm] if ptm in num_conserved_transcripts else 0
            num_lost = num_lost_transcripts[ptm] if ptm in num_lost_transcripts else 0
            #check if there are any conserved transcripts (or if not and is NaN)
            if num_conserved != num_conserved and num_lost == num_lost:
                conservation_score.append(0)
            elif num_conserved == 0 and num_lost == 0:
                conservation_score.append(1)
            elif num_conserved != num_conserved and num_lost != num_lost:
                conservation_score.append(1)
            #check if any lost transcripts: if not replace NaN with 0 when calculating
            elif num_lost != num_lost and num_conserved == num_conserved:
                conservation_score.append(1)
            else:
                conservation_score.append(num_conserved/(num_conserved+num_lost))
        if log_run:
            logger.info(f"Saving conservation scores to PTM_info dataframe in '{save_col}' column")
        self.ptm_info[save_col] = conservation_score
        if return_score:
            if unique_isoforms:
                num_isoforms = alt_ptms['Isoform ID'].nunique()
            else:
                num_isoforms = alt_ptms['Alternative Transcript'].nunique()
            return self.ptm_info[self.ptm_info[save_col] == 1].shape[0]/self.ptm_info.shape[0], num_isoforms
    
    def compareAllFlankSeqs(self, flank_size = 5, unique_isoforms = True):
        """
        Given the alternative ptms and canonical ptm data, compare flanking sequences and determine if they are identical. 
        
        Parameters
        ----------
        mapper: PTM_mapper object
            mapper object containing flanking sequence data for both alternative(alternative_ptms or isoform_ptms) and canonical ptms (ptm_info)
        flank_size:
            size of the flanking sequence to compare. IMPORTANT, this should not be larger than the available flanking sequence in the ptm_info dataframe
        unique_isoforms: bool
            If True, compare ptms found in unique isoforms only. If False, compare ptms found in all transcripts, regardless of if there is redundant protein sequences
        
        Returns
        -------
        conserved_flank: list
            list of 1s and 0s indicating if the flanking sequence is conserved (1) or not (0)
        """
        conserved_flank = []
        if unique_isoforms:
            res = self.isoform_ptms.copy()
        else:
            res = self.alternative_ptms.copy()
        for i in res.index:
            #check if alt flanking seq exists
            alt_flank = res.loc[i, 'Flanking Sequence']
            if alt_flank != alt_flank:
                conserved_flank.append(np.nan)
            else:
                #find mod loc in sequence
                mod_loc = int(re.search('[a-z]+', alt_flank).span()[0])
                #find n_term loc (in case flank is longer than actual surrounding region)
                if mod_loc - flank_size >= 0:
                    n_term = mod_loc -flank_size
                else:
                    n_term = 0
                
                #find c_term loc (in case flank is longer than actual surrounding region)
                if mod_loc + flank_size+1 <= len(alt_flank):
                    c_term = mod_loc + flank_size+1
                alt_flank = alt_flank[n_term:c_term]

                #check if flank sequence exists/was found and save
                if alt_flank != alt_flank:
                    conserved_flank.append(np.nan)
                else:
                    ptm = res.loc[i, 'Source of PTM']
                    if ';' in ptm:
                        ptm = ptm.split(';')[0]
                    can_flank = self.ptm_info.loc[ptm, 'Flanking Sequence']
                    can_flank = can_flank[n_term:c_term]
                    conserved_flank.append(matchedFlankSeq(can_flank, alt_flank))
        return conserved_flank
    
    def compareAllTrypticFragments(self, unique_isoforms = True):
        """
        Given the alternative ptms and canonical ptm data, compare flanking sequences and determine if they are identical. 
        
        Parameters
        ----------
        flank_size:
            size of the flanking sequence to compare. IMPORTANT, this should not be larger than the available flanking sequence in the ptm_info dataframe
        """
        conserved_frag = []
        if unique_isoforms:
            res = self.isoform_ptms.copy()
        else:
            res = self.alternative_ptms.copy()
        for i in tqdm(res.index):
            #check if alt flanking seq exists
            alt_frag = res.loc[i, 'Tryptic Fragment']
            #check if flank sequence exists/was found and save
            if alt_frag != alt_frag:
                conserved_frag.append(np.nan)
            else:
                ptm = res.loc[i, 'Source of PTM']
                if ';' in ptm:
                    ptm = ptm.split(';')[0]
                can_frag = self.ptm_info.loc[ptm, 'Tryptic Fragment']
                conserved_frag.append(matchedFlankSeq(can_frag, alt_frag))
        return conserved_frag
    
    def getFlankConservation(self, flank_size = 5, unique_isoforms = True):
        """
        For each PTM, compare the flanking sequence of the PTM in the canonical isoform to the flanking sequence of the PTM in the alternative isoform. If the flanking sequence is identical, the PTM is considered to be conserved (1). 

        Parameters
        ----------
        flank_size : int, optional
            Size of the flanking sequence to compare. IMPORTANT, this should not be larger than the available flanking sequence in the ptm_info or alternative_ptms/isoform_ptms dataframe, otherwise it will return potentially incorrect results. The default is 5.
        unique_isoforms : bool, optional
            If True, compare ptms found in unique isoforms only. If False, compare ptms found in all transcripts, regardless of if there is redundant protein sequences. The default is True.  

        Returns
        -------
        None, but saves results to PTM_mapper object
        """
        if unique_isoforms:
            self.isoform_ptms['Conserved Flank'] = self.compareAllFlankSeqs(flank_size= flank_size, unique_isoforms = unique_isoforms)
        else:
            self.alternative_ptms['Conserved Flank'] = self.compareAllFlankSeqs(flank_size=flank_size, unique_isoforms = unique_isoforms)

    def addSpliceEventsToAlternative(self, splice_events_df):
        """
        Adds splice event information obtained from get_splice_events.py to alternative_ptms dataframe

        Parameters
        ----------
        splice_events_df : pandas dataframe
            dataframe containing splice event information

        Returns
        -------
        Updated version of alternative_ptm dataframe with splice event information added
        """
        logger.info('Adding splice event information to alternative ptm dataframe')
        #save size of original dataframe
        original_size = self.alternative_ptms.shape[0]

        #extract only the necessary columns from splice_events_df
        splice_events_df = splice_events_df[['Exon ID (Canonical)', 'Exon ID (Alternative)', 'Alternative Transcript', 'Event Type']].drop_duplicates()
        alternative_ptms = self.alternative_ptms.copy()
        if 'Exon ID (Alternative)' in alternative_ptms.columns:
            alternative_ptms = alternative_ptms.drop(columns = 'Exon ID (Alternative)')

        if 'Event Type' in alternative_ptms.columns:
            alternative_ptms = alternative_ptms.drop(columns = 'Event Type')


        #separate each exon into its own row, and merge with splice event information
        alternative_ptms['Exon ID (Canonical)'] = alternative_ptms['Exon ID (Canonical)'].apply(lambda x: x.split(';'))
        alternative_ptms = alternative_ptms.explode('Exon ID (Canonical)').drop_duplicates()
        alternative_ptms = alternative_ptms.merge(splice_events_df, on = ['Exon ID (Canonical)', 'Alternative Transcript'], how = 'left')
        
        

        exploded_ptms = self.explode_PTMinfo()
        exploded_ptms = exploded_ptms[exploded_ptms['Isoform Type'] == 'Canonical']
        #check PTMs in mutually exclusive exons to see if they might be conserved, add conservation data if so (PTM location in isoform, etc.)
        mxe_ptm_candidates = alternative_ptms[alternative_ptms['Event Type'] == 'Mutually Exclusive'].copy()
        alternative_ptms = alternative_ptms[alternative_ptms['Event Type'] != 'Mutually Exclusive']
        alt_residue_list = []
        alt_position_list = []
        for i, row in mxe_ptm_candidates.iterrows():
            if not utility.stringToBoolean(row['Ragged']):
                ptm = row['Source of PTM']
                
                #get canonical exon info
                canonical_exon_id = row['Exon ID (Canonical)']
                ptm_info_of_interest = exploded_ptms.loc[(exploded_ptms['PTM'].isin(ptm.split(';'))) & (exploded_ptms['Exon stable ID'] == canonical_exon_id)].iloc[0]
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
                
        #add alternative residue/position info to mxe_ptm_candidates
        mxe_ptm_candidates['Alternative Residue'] = alt_residue_list
        mxe_ptm_candidates['Alternative Protein Position (AA)'] = alt_position_list
        
        #replace old information for mxe ptm candidates with new info
        mxe_ptm_candidates['Mapping Result'] = mxe_ptm_candidates.apply(lambda x: 'Success' if x['Alternative Residue'] == x['Alternative Residue'] else 'Not Found', axis = 1)
        alternative_ptms = pd.concat([alternative_ptms, mxe_ptm_candidates])
        

        #annotate alt ptms that are from transcripts associated with canonical isoform
        canonical_transcripts = config.translator.loc[config.translator['UniProt Isoform Type'] == 'Canonical', 'Transcript stable ID'].unique()
        alternative_ptms.loc[alternative_ptms['Alternative Transcript'].isin(canonical_transcripts), 'Event Type'] = 'Canonical'
        alternative_ptms.loc[(alternative_ptms['Alternative Transcript'].isin(canonical_transcripts)), 'Exon ID (Alternative)'] = alternative_ptms.loc[(alternative_ptms['Alternative Transcript'].isin(canonical_transcripts)), 'Exon ID (Canonical)']


        #collapse into rows for matching event types and exon ids so that each PTM is now a unique row
        alternative_ptms = alternative_ptms.drop_duplicates()
        cols = [col for col in alternative_ptms.columns if col != 'Event Type' and col != 'Exon ID (Canonical)' and col != 'Exon ID (Alternative)']
        #alternative_ptms = alternative_ptms.replace(np.nan, 'nan')
        alternative_ptms = alternative_ptms.groupby(cols, dropna = False).agg(utility.join_entries).reset_index()
        #alternative_ptms = alternative_ptms.replace('nan', np.nan)

        conflicting_events = alternative_ptms[alternative_ptms.duplicated(subset = ['Gene stable ID', 'Alternative Transcript', 'Source of PTM', 'Genomic Coordinates'], keep = False)].sort_values(by = 'Source of PTM')

        #if there are conflicting events, solve those conflicts
        if conflicting_events.shape[0] >= 1:
            #if on of events is successful, grab that one
            if 'Success' in conflicting_events['Mapping Result'].values:
                conflicting_events = conflicting_events[conflicting_events['Mapping Result'] == 'Success']
            else:
                conflicting_events = conflicting_events.iloc[0]

            nonconflicting_events = alternative_ptms[~alternative_ptms.duplicated(subset = ['Gene stable ID', 'Alternative Transcript', 'Source of PTM', 'Genomic Coordinates'], keep = False)].sort_values(by = 'Source of PTM')
            alternative_ptms = pd.concat([conflicting_events, nonconflicting_events])

        #make sure alternative ptm dataframe is the same size as before
        if alternative_ptms.shape[0] != original_size:
            logger.error('Failed to add splice event information to alternative ptm dataframe, due to unexpected change in dataframe size')
            raise ValueError('Alternative PTM dataframe is not the same size as before. Something went wrong during processing, please make sure no duplicates or other errors exist in splice event or alternative ptm dataframe')
        else:
            self.alternative_ptms = alternative_ptms.copy()
            logger.info('Successfully added splice event information to alternative ptm dataframe')
        

    def annotateAlternativePTMs(self):
        """
        Annotates alternative_ptms dataframe obtained from self.mapPTMsToAlternativeExons() with additional layers of information, including flanking sequences, associated splice events, and TRIFID functional scores associated with the transcript

        Parameters
        ----------
        None

        Returns
        -------
        Updated alternative_ptms dataframe
        """
        logger.info('Adding additional context to alternative PTMs dataframe: see following log messages for details.')
        if os.path.exists(config.processed_data_dir + 'splice_events.csv') and 'Event Type' not in self.alternative_ptms.columns:
            logger.info('Adding splice events responsible for any potential changes to PTMs')
            print('Adding splice events to alternative dataframes and checking MXE events for conserved PTMs')
            sevents = pd.read_csv(config.processed_data_dir + 'splice_events.csv')
            self.addSpliceEventsToAlternative(sevents)
            

        if 'TRIFID Score' in self.transcripts.columns and 'TRIFID Score' not in self.alternative_ptms.columns:
            logger.info('Adding TRIFID scores downloaded from APPRIS')
            self.alternative_ptms = self.alternative_ptms.merge(self.transcripts['TRIFID Score'], right_index = True, left_on = 'Alternative Transcript', how = 'left')

            

        print('Getting flanking sequences around PTMs in alternative isoforms')
        logger.info('Getting flanking sequence around PTMs in alternative isoforms')
        flank = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Position (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                flank.append(np.nan)
            else:
                flank.append(self.getFlankingSeq(int(pos), transcript_id, flank_size = 10))
        self.alternative_ptms['Flanking Sequence'] = flank
        

        print('Getting tryptic fragments that include each PTM in alternative isoforms')
        logger.info('Getting tryptic fragments that include each PTM in alternative isoforms')
        tryptic = []
        for pos, transcript_id in zip(self.alternative_ptms['Alternative Protein Position (AA)'], self.alternative_ptms['Alternative Transcript']):
            if pos != pos:
                tryptic.append(np.nan)
            else:
                tryptic.append(self.getTrypticFragment(int(pos), transcript_id))
        self.alternative_ptms['Tryptic Fragment'] = tryptic

            

    def getIsoformSpecificPTMs(self, required_length = 20):
        """
        Reduce alternative ptm dataframe to ptms that are unique to a specific protein sequence, rather than a specific transcript. This avoids issue in which multiple transcripts can code for the same protein isoform.

        Parameters
        ----------
        required_length : int, optional
            Minimum length of protein isoform to be considered. The default is 20 amino acids.

        Returns
        -------
        isoform_ptms (as attribute of mapper object): dataframe
            Dataframe of PTMs that are unique to a specific protein isoform, rather than a specific transcript.
        """
        #get isoform data, then separate isoform data into unique rows for each transcript
        isoforms = self.isoforms.copy()
        isoforms = isoforms[['Isoform ID', 'Isoform Type', 'Transcript stable ID', 'Isoform Length']]
        isoforms['Transcript stable ID'] = isoforms['Transcript stable ID'].apply(lambda x: x.split(';'))
        isoforms = isoforms.explode('Transcript stable ID')
        isoforms = isoforms.rename({'Transcript stable ID': 'Alternative Transcript'}, axis = 1)

        #merge isoform and alternative ptm information
        isoform_ptms = self.alternative_ptms.merge(isoforms, on = 'Alternative Transcript', how = 'left').copy()

        #collapse on isoform ID and PTM
        #cols_to_collapse = ['Gene stable ID', 'Alternative Transcript',  'Exon ID (Alternative)', 'Exon ID (Canonical)', 'Second Exon']
        #if 'TRIFID Score' in isoform_ptms.columns:
        #    cols_to_collapse.append('TRIFID Score')
        #collapse rows on ptm and isoform id
        #group_cols = [col for col in isoform_ptms.columns if col not in cols_to_collapse]
        #aggregate columns into single string (remove nans and convert numeric values from TRIFID to string)
        #isoform_ptms = isoform_ptms.groupby(group_cols).agg(lambda x: ';'.join(np.unique([str(y) for y in x if y == y]))).reset_index()
        #convert empty strings to np.nan
        #isoform_ptms = isoform_ptms.replace('',np.nan)

        #drop duplicate rows by isoform id
        isoform_ptms = isoform_ptms.drop_duplicates(subset = ['Isoform ID', 'Source of PTM'])
        isoform_ptms = isoform_ptms.drop('Alternative Transcript', axis = 1)

        #drop isoforms shorter than 20 amino acids
        isoform_ptms = isoform_ptms[isoform_ptms['Isoform Length'] >= required_length]

        self.isoform_ptms = isoform_ptms

    def transcriptToIsoform(self, transcript_list):
        """
        Given a list of transcripts, convert those transcript IDs to their associated isoform IDs generated by the ExonPTMapper object.

        Parameters
        ----------
        mapper: PTMapper object
            Mapper object containing information about PTMs and isoforms
        transcript_list: list
            List of Ensembl transcript IDs to convert to isoform IDs
        
        Returns
        -------
        isoform_ids: list
            List of isoform IDs associated with the input transcript IDs
        
        """
        isoform_ids = []
        for i,row in tqdm(self.isoforms.iterrows(), total = self.isoforms.shape[0]):
            transcripts = row['Transcript stable ID'].split(';')
            for trans in transcripts:
                if trans in transcript_list:
                    isoform_ids.append(row['Isoform ID'])
        return np.unique(isoform_ids)

    def save_mapper(self, to_tsv = True, to_pickle = False):
        """
        Save mapper object to .tsv files and/or pickle file. By default, only saves to .tsv files, due to potential security concerns of pickles

        Parameters
        ----------
        to_tsv: bool, optional
            Whether to save mapper attributes to .tsv files. The default is True.
        to_pickle: bool, optional
            Whether to save mapper object as a pickle file. The default is False.

        Returns
        -------
        None
        """
        if to_pickle:
            print('saving pickled mapper object')
            self.translator = config.translator.copy()
            with open(config.processed_data_dir + 'mapper.p', 'wb') as f:
                pickle.dump(self.__dict__, f)
        
        if to_tsv:
            print('Saving mapper attributes to .tsv files')
            if self.genes is not None:
                self.genes.to_csv(config.processed_data_dir + 'genes.csv')
            
            if self.transcripts is not None:
                self.transcripts.to_csv(config.processed_data_dir + 'transcripts.csv')
            
            if self.exons is not None:
                self.exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)

            if self.ptm_info is not None:
                self.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
            
            if self.ptm_coordinates is not None:
                self.ptm_coordinates.to_csv(config.processed_data_dir + 'ptm_coordinates.csv')

            if self.isoforms is not None:
                self.isoforms.to_csv(config.processed_data_dir + 'isoforms.csv')

            if self.alternative_ptms is not None:
                self.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv')


            
            

        
    def load_PTMmapper(self, from_pickle = False):
        """
        Load all data files from processed data directory listed in the config file, and save as attributes of the PTM_mapper object. __init__ function calls this function

        Parameters
        ----------
        from_pickle : bool, optional
            Whether to load the mapper object via pickle, if it exists. The default is False, and will manually load the mapper object through each .tsv file in processed data directory.

        Returns
        -------
        new attributes of the PTM_mapper object, depending on what files were found (exons, transcripts, genes, ptm_info, ptm_coordinates, alternative_ptms, isoforms)
        """
        #check if each data file exists: if it does, load into mapper object
        if from_pickle and os.path.isfile(config.processed_data_dir + 'mapper.p'):
            print('Loading pickled mapper object')
            with open(config.processed_data_dir + 'mapper.p', 'rb') as f:
                self.__dict__ = pickle.load(f)
        else:
            if from_pickle:
                print('No pickled mapper object found, loading directly from tsv files')
            else:
                print('Loading mapper object from .tsv files')
        
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
                self.ptm_info = pd.read_csv(config.processed_data_dir + 'ptm_info.csv',index_col = 0, dtype = {'PTM Location (AA)':int})
                
                #check to see if ptm_info is collapsed or not (if each row is a unique ptm)
                if len(np.unique(self.ptm_info.index)) != self.ptm_info.shape[0]:
                    self.ptm_info = self.ptm_info.reset_index()
                    self.ptm_info = self.ptm_info.rename({'index':'PTM'}, axis = 1)
            else:
                self.ptm_info = None
                
            if os.path.exists(config.processed_data_dir + 'ptm_coordinates.csv'):
                print('Loading genomic coordinates of PTMs associated with canonical proteins')
                self.ptm_coordinates = pd.read_csv(config.processed_data_dir + 'ptm_coordinates.csv',index_col = 0,
                                                    dtype = {'Source of PTM': str, 'Chromosome/scaffold name': str, 'Gene Location':int,
                                                    'Ragged': str, 'PTM Position in Canonical Isoform':str})
            else:
                self.ptm_coordinates = None
                
            if os.path.exists(config.processed_data_dir + 'alternative_ptms.csv'):
                print('Loading information on PTMs projected onto variant proteins')
                self.alternative_ptms = pd.read_csv(config.processed_data_dir + 'alternative_ptms.csv', dtype = {'Exon ID (Alternative)':str, 'Chromosome/scaffold name': str, 'Ragged':str, 'Genomic Coordinates':str, 'Second Exon': str, 'Alternative Residue': str, 'Protein':str})
            else:
                self.alternative_ptms = None
                
            if os.path.exists(config.processed_data_dir + 'isoform_ptms.csv'):
                print('Loading information on PTMs on unique protein isoforms')
                self.isoform_ptms = pd.read_csv(config.processed_data_dir + 'alternative_ptms.csv', dtype = {'Exon ID (Alternative)':str, 'Chromosome/scaffold name': str, 'Ragged':str, 'Genomic Coordinates':str, 'Second Exon': str, 'Alternative Residue': str, 'Protein':str})
            else:
                self.isoform_ptms = None

               
    
def RNA_to_Prot(pos, cds_start):
    """
    Given nucleotide location, indicate the location of residue in protein

    Parameters
    ----------
    pos : int
        nucleotide position in transcript (assumes that provided position indicates the first base pair of the codon of interest)
    cds_start : int
        location of first base pair of the coding sequence in transcript

    Returns
    -------
    Location of residue of interest, within the protein
    """
    return (int(pos) - int(cds_start)+3)/3

def Prot_to_RNA(pos, cds_start):
    """
    Given residue location, indicate the location of first nucleotide in transcript (first nucleotide in the codon)

    Parameters
    ----------
    pos : int
        residue position in protein (1 is the first residue)
    cds_start : int
        location of first base pair of the coding sequence in transcript 
    
    Returns
    -------
    Location of first nucleotide coding for the residue of interest, within the transcript
    """
    return (int(pos)*3 -3) + int(cds_start)
    
def getDistanceToBoundary(exon, gene_loc, strand):
    """
    Given an exon, gene location, and strand, identify the distance of the gene location to the boundary of the exon. Used by MapPTMs_all() function.

    Parameters
    ----------
    exon : pandas series
        row from from PTM_mapper.exons dataframe
    gene_loc : int
        genomic location of PTM
    strand : int
        strand of gene where PTM is located. 1 is forward strand, -1 is reverse strand.

    Returns
    -------
    distance_to_bound : int
        distance of PTM to the boundary of the exon, in base pairs
    """
    if strand == 1:
        distance_to_bound = exon['Exon End (Gene)'] - gene_loc
    else:
        distance_to_bound = gene_loc - exon['Exon Start (Gene)']
    return distance_to_bound
    
def getGenomicCoordinates(chromosome, gene_loc, strand):
    """
    Given the gneomic location of a non-ragged PTM, return the genomic coordinates formatted as a string. Used by MapPTMs_all() function.

    Parameters
    ----------
    chromosome : int
        chromosome number of gene
    gene_loc : int
        genomic location of PTM
    strand : int
        strand of gene where PTM is located

    Returns
    -------
    genomic coordinates of PTM, formatted as a string
    """
    if strand == 1:
        return f"chr{chromosome}:{gene_loc}-{gene_loc+2}:{strand}"
    elif strand == -1:
        return f'chr{chromosome}:{gene_loc-2}-{gene_loc}:{strand}'
        
def getRaggedCoordinates(chromosome, gene_loc, ragged_loc, distance_to_bound, strand):
    """
    Given the two genomic locations of a ragged PTM, return the genomic coordinates formatted as a string. Used by MapPTMs_all() function.

    Parameters
    ----------
    chromosome : int
        chromosome number of gene
    gene_loc : int
        first genomic location within gene (smaller of the two)
    ragged_loc : int
        second genomic location within gene (larger of the two)
    distance_to_bound : int
        distance of PTM to the boundary of the exon
    strand : int
        strand of gene where PTM is located
    
    Returns
    -------
    genomic coordinates of PTM, formatted as a string
    
    """
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
    """
    Given an exon, mapper object, and genomic location to check, identify the frame, residue, and amino acid position associated with that genomic location. Used by mapPTMstoAlternative() function.

    Parameters
    ----------
    exon : pandas series
        row from from PTM_mapper.exons dataframe
    mapper : PTM_mapper object
        mapper object containing transcript and exon information
    gene_loc : int
        genomic location to check
    strand : int
        strand of gene where exon is located

    Returns
    -------
    frame : int
        frame associated with genomic location. Can be 1, 2, or 3, where 1 indicates that the location is in frame, and 1 or 2 indicate that the location is out of frame (i.e not the first base pair of a codon).
    residue : str
        amino acid residue associated with genomic location. If frame is not 1, residue will be NaN
    aa_pos : int
        amino acid position associated with genomic location. If frame is not 1, aa_pos will be NaN
    """
    #make sure transcript associated with exon contains coding information
    transcript = mapper.transcripts.loc[exon['Transcript stable ID']]
    if transcript['Relative CDS Start (bp)'] == transcript['Relative CDS Start (bp)']:
        #check frame: if in frame, return residue and position
        frame, residue, aa_pos = utility.checkFrame(exon, transcript, gene_loc, 
                                 loc_type = 'Gene', strand = strand, return_residue = True)
        return frame, residue, aa_pos
    else:
        return np.nan, np.nan, np.nan

def convert_genomic_coordinates(location, chromosome, strand, from_type = 'hg38', to_type = 'hg19', liftover_object = None):
    if liftover_object is None:
        liftover_object = pyliftover(from_type,to_type)
    
    if strand == 1:
        strand = '+'
    else:
        strand = '-'
    
    chromosome = f'chr{chromosome}'
    results = liftover_object.convert_coordinate(chromosome, location - 1, strand)
    if len(results) > 0:
        new_chromosome = results[0][0]
        new_strand = results[0][2]
        if new_chromosome == chromosome and new_strand == strand:
            return int(results[0][1]) + 1
        else:
            return -1
    else:
        return np.nan

def convertToHG19(hg38_location, chromosome, strand, liftover_object = None):
    """
    Use pyliftover to convert from hg38 to hg19 coordinates systems
    
    Parameters
    ----------
    hg38_location: int
        genomic location to convert, in hg38 version of Ensembl
    chromosome: int
        chromosome number of genomic location
    strand: int
        strand of gene associated with genomic location
    liftover_object: pyliftover object
        object used to convert coordinates. If None, will create new object
        
    Returns
    -------
    hg19_coordinates: int
        genomic coordinates in hg19 version of Ensembl
    """
    if liftover_object is None:
        liftover_object = pyliftover('hg38','hg19')
    
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
            return int(results[0][1]) + 1
        else:
            return -1
    else:
        return np.nan
    
def matchedFlankSeq(seq1, seq2):
    """
    Given two sequences (or strings), determine if they are identical.

    Parameters
    ----------
    seq1, seq2: str
        sequences to compare (order does not matter)
    
    Returns
    -------
    1 if the sequences are identical, 0 otherwise
    """
    return (seq1 == seq2)*1
        
        
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

    

def run_mapping(phosphositeplus_file = None, restart_all = False, restart_mapping = False, exon_sequences_fname = 'exon_sequences.fasta.gz',
                coding_sequences_fname = 'coding_sequences.fasta.gz', trifid_fname = 'APPRIS_functionalscores.txt'):
    
    """
    Run the complete mapping process, starting from downloading data from Ensembl, all the way to mapping PTMs to alternative exons. Will only run steps that have not already been completed (based on data downloaded from processed_data_dir), unless either restart_all (repeat all steps) or restart_mapping (repeat mapping steps but not processing of ensemble data) are set to True. 

    Parameters
    ----------
    restart_all : bool, optional
        Whether to restart the entire mapping process. The default is False.
    restart_mapping : bool, optional
        Whether to restart the mapping process, but not the processing of ensemble data. The default is False. If restart_all is True, than restart_mapping will automatically be True
    exon_sequences_fname : str, optional
        Name of exon sequence file downloaded from Ensembl. The default is 'exon_sequences.fasta.gz'. This file should be located in the source_data_dir
    coding_sequences_fname : str, optional
        Name of coding sequence file downloaded from Ensembl. The default is 'coding_sequences.fasta.gz'. This file should be located in the source_data_dir
    trifid_fname : str, optional
        Name of TRIFID functional score file downloaded from APPRIS. The default is 'APPRIS_functionalscores.txt'. This file should be located in the source_data_dir. If not provided, then TRIFID functional scores will not be added to the transcripts dataframe.

    Returns
    -------
    mapper : PTM_mapper object 
        mapper object containing all information about PTMs and alternative exons. Will also save this information as .csv files in processed_data_dir 
    """
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
            
    #get protein sequence associated with each exon
    if 'Exon AA Seq (Full Codon)' not in mapper.exons.columns or restart_all:
        print('Getting exon-specific amino acid sequence\n')
        mapper.exons = processing.getAllExonSequences(mapper.exons, mapper.transcripts, mapper.genes)
        mapper.exons.to_csv(config.processed_data_dir + 'exons.csv', index = False)
        
    #identify canonical transcripts with matching protein sequence information in Ensembl and ProteomeScout
    if config.available_transcripts is None:
        print('Identifying transcripts with matching information from UniProt canonical proteins in ProteomeScout')
        if phosphositeplus_file is None:
            phosphositeplus_data = None
        else:
            phosphositeplus_data = phosphositeplus_data = pd.read_csv(phosphositeplus_file, index_col = 0)
        processing.getMatchedTranscripts(mapper.transcripts, phosphosite_data = phosphositeplus_data, update = restart_all)
    
    #get protein-specific information
    if mapper.proteins is None or restart_all:
        print('Getting protein-specific information')
        mapper.proteins = processing.getProteinInfo(mapper.genes)
        print('saving\n')
        mapper.proteins.to_csv(config.processed_data_dir + 'proteins.csv')
        
    #collapse transcripts sharing the same protein sequence into a single row to create isoform datframe
    if mapper.isoforms is None or restart_all:
        print('Getting unique protein isoforms from transcript data')
        mapper.isoforms = processing.getIsoformInfo(mapper.transcripts)
        print('saving\n')
        mapper.isoforms.to_csv(config.processed_data_dir + 'isoforms.csv', index = False)

    #if restarting the whole process, then set the restart_mapping variable to true
    if restart_all:
        restart_mapping = True
        
    #get PTMs associated with canonical proteins from UniProt with matching transcripts in Ensembl
    if mapper.ptm_info is None or restart_mapping:
        mapper.find_ptms_all(phosphositeplus_file = phosphositeplus_file, collapse = True)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        
    if mapper.ptm_coordinates is None or restart_mapping:
        print('Mapping PTMs to their genomic location')
        mapper.mapPTMs_all(restart = restart_mapping)
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        mapper.ptm_coordinates.to_csv(config.processed_data_dir + 'ptm_coordinates.csv')
        mapper.proteins.to_csv(config.processed_data_dir + 'proteins.csv')

    
    ####run additional analysis
    if 'Tryptic Fragment' not in mapper.ptm_info.columns or restart_mapping:
        logger.info('Getting tryptic fragments associated with canonical ptms')
        mapper.getAllTrypticFragments()
    if 'Flanking Sequence' not in mapper.ptm_info.columns or restart_mapping:
        logger.info('Getting flanking sequences associated with canonical ptms')
        mapper.getAllFlankingSeqs(flank_size = 10)
    if 'inDomain' not in mapper.ptm_info.columns or restart_mapping:
        logger.info('Identifying PTMs that are in protein domains')
        mapper.findAllinDomains()
        #report the fraction of ptms in domains
        logger.info('Fraction of PTMs in domains: ' + str(mapper.ptm_info[mapper.ptm_info['inDomain']].shape[0]/mapper.ptm_info.shape[0]))
    print('saving\n')
    mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
    
    if mapper.alternative_ptms is None:
        print('Mapping PTM sites onto alternative isoforms')
        logger.info('Mapping PTM sites onto alternative isoforms')
        mapper.projectPTMs_toIsoformExons()
        mapper.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv', index = False)
        print('saving\n')
   
        
    if not os.path.exists(config.processed_data_dir + 'splice_events.csv'):
        print('Identify splice events that result in alternative isoforms')
        logger.info("Identifying splice events that result in alternative isoforms")
        splice_events_df = get_splice_events.identifySpliceEvents_All(mapper.exons, mapper.proteins, mapper.transcripts, mapper.genes)
        print('saving\n')
        splice_events_df.to_csv(config.processed_data_dir + 'splice_events.csv', index = False)
        
    if 'Tryptic Fragment' not in mapper.alternative_ptms.columns:
        print('Annotate PTM sites on alternative isoforms')
        logger.info('Annotating PTM sites on alternative isoforms')
        mapper.annotateAlternativePTMs()
        print('saving\n')
        mapper.alternative_ptms.to_csv(config.processed_data_dir + 'alternative_ptms.csv', index = False)
        
    if 'Number of Conserved Transcripts' not in mapper.ptm_info.columns:
        print('Calculating conservation of each PTM across alternative transcripts')
        logger.info('Calculating PTM conservation across alternative transcripts')
        mapper.calculate_PTMconservation()
        print('saving\n')
        mapper.ptm_info.to_csv(config.processed_data_dir + 'ptm_info.csv')
        
    return mapper