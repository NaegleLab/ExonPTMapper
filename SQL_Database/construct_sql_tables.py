import os
import sqlite3
import pandas as pd

import sys
sys.path.append('../')
from ExonPTMapper import config, plot

mapper = plot.plotter()

def get_tables(conn):
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def construct_gene_table(conn, mapper):
    tables = get_tables(conn)
    genes = mapper.genes.copy()
    #set up gene table columns and keys
    if 'genes' in tables:
        conn.execute("""DROP TABLE genes""")
    create_table_query = f"""CREATE TABLE genes (
        Gene_stable_ID TEXT PRIMARY KEY,
        Gene_name TEXT,
        Strand INT, 
        Chromosome TEXT,
        Gene_start INT,
        Gene_end INT,
        Number_of_Uniprot_Proteins INT
    );"""
    #construct mapper table
    conn.execute(create_table_query)

    #add data to table
    columns_to_drop = ['Associated Uniprot Proteins']
    columns_to_keep = [col for col in genes.columns if col not in columns_to_drop]

    genes = genes[columns_to_keep].drop_duplicates().copy()
    genes = genes.rename(columns={'Gene stable ID': 'Gene_stable_ID', 
                            'Gene name': 'Gene_name', 
                            'Strand': 'Strand', 
                            'Chromosome/scaffold name': 'Chromosome', 
                            'Gene start (bp)': 'Gene_start', 
                            'Gene end (bp)': 'Gene_end', 
                            'Number of Associated Uniprot Proteins': 'Number_of_Uniprot_Proteins'})
    genes.to_sql('genes', conn, if_exists='append', index=True, index_label='Gene_stable_ID')



def construct_protein_tables(conn, mapper):
    #check what proteins are currently present in the table
    tables = get_tables(conn)
    if 'proteins' in tables:
        conn.execute("""DROP TABLE proteins""")
    if 'protein_transcript' in tables:
        conn.execute("""DROP TABLE protein_transcript""")
    create_table_query = f"""CREATE TABLE proteins (
        UniProtKB_ID TEXT PRIMARY KEY,
        SwissProt_ID TEXT,
        UniProt_Isoform_Type TEXT,
        Unique_gene TEXT,
        Number_of_PTMs INTEGER
    );"""
    #construct mapper table
    conn.execute(create_table_query)

    proteins = mapper.proteins[['UniProtKB/Swiss-Prot ID', 'UniProt Isoform Type', 'Unique Gene', 'Number of PTMs']].copy()
    proteins = proteins.rename(columns={ 
                            'UniProtKB/Swiss-Prot ID': 'SwissProt_ID', 
                            'UniProt Isoform Type': 'UniProt_Isoform_Type', 
                            'Unique Gene': 'Unique_gene', 
                            'Number of PTMs': 'Number_of_PTMs'})

    proteins.to_sql('proteins', conn, if_exists='append', index=True, index_label='UniProtKB_ID')


    #### Gene to proteins ####
    if 'gene_to_proteins' in tables:
        conn.execute("""DROP TABLE gene_to_proteins""")

    create_table_query = f"""CREATE TABLE gene_to_proteins (
        SwissProt_ID TEXT,
        Gene_stable_ID TEXT,
        Gene_name TEXT,
        PRIMARY KEY (SwissProt_ID, Gene_stable_ID),
        FOREIGN KEY (Gene_stable_ID) REFERENCES genes(Gene_stable_ID)
    );"""
    #construct mapper table
    conn.execute(create_table_query)

    #extract connections between genes and proteins
    translator = config.translator[['Gene stable ID', 'Gene name', 'UniProtKB/Swiss-Prot ID']].dropna().drop_duplicates().copy()
    translator = translator.rename(columns={'Gene stable ID': 'Gene_stable_ID', 'Gene name': 'Gene_name', 'UniProtKB/Swiss-Prot ID': 'SwissProt_ID'})

    #remove genes or isoform ids not in parent tables
    translator = translator[translator['Gene_stable_ID'].isin(mapper.genes.index)]
    translator.to_sql('gene_to_proteins', conn, if_exists='append', index=False)

    #set an index on Gene_stable_ID
    query = """CREATE INDEX idx_gene_stable_id ON gene_to_proteins (Gene_stable_ID);"""
    conn.execute(query)

def construct_domains_table(conn, mapper):
    #get all swissprot ids
    swissprot = mapper.proteins['UniProtKB/Swiss-Prot ID'].unique()

    #get all protein domain information
    domain_list = []
    for prot in swissprot:
        domains = config.ps_api.get_domains(prot, domain_type = 'pfam')
        #iterate through and add protein to tuple
        if domains != -1:
            domains = [[prot] + list(d) for d in domains]
            domain_list.extend(domains)

    domains = pd.DataFrame(domain_list, columns = ['SwissProt_ID', 'Domain_type', 'Domain_start', 'Domain_end'])

    tables = [table[0] for table in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    if 'domains' in tables:
        conn.execute("""DROP TABLE domains""")
    create_table_query = """CREATE TABLE domains (
        SwissProt_ID TEXT,
        Domain_type TEXT,
        Domain_start INT,
        Domain_end INT
    );"""

    conn.execute(create_table_query)
    domains.to_sql('domains', conn, if_exists='append', index=False)

def construct_isoform_tables(conn, mapper):

    ### isoform table ###
    tables = get_tables(conn)

    #check to make gene table is created
    if 'genes' not in tables:
        raise ValueError("Gene table is not created but is needed for isoform table. Run `construct_gene_tables` first.")

    #remove isoform table if exists and any tables that require that info
    if 'isoform_ptms' in tables:
        conn.execute("""DROP TABLE isoform_ptms""")
    if 'isoform_to_transcript' in tables:
        conn.execute("""DROP TABLE isoform_to_transcript""")
    if 'isoforms' in tables:
        conn.execute("""DROP TABLE isoforms""")

    #construct table query
    create_table_query = """CREATE TABLE isoforms (
        Isoform_ID TEXT PRIMARY KEY,
        Isoform_source TEXT,
        Gene_stable_ID TEXT,
        Isoform_Type TEXT,
        Isoform_Length INT,
        Sequence TEXT,
        FOREIGN KEY (Gene_stable_ID) REFERENCES genes(Gene_stable_ID)
    );"""

    conn.execute(create_table_query)

    isoforms = mapper.isoforms.copy()
    isoforms = isoforms.drop(columns = ['Transcript stable ID']).drop_duplicates()
    isoforms = isoforms.rename(columns = {'Gene stable ID':'Gene_stable_ID', 'Amino Acid Sequence':'Sequence', 'Isoform ID':'Isoform_ID', 'Isoform Type':'Isoform_Type', 'Isoform Length':'Isoform_Length'})
    isoforms = isoforms.drop_duplicates(subset = 'Isoform_ID')
    isoforms['Isoform_Source'] = isoforms['Isoform_ID'].apply(lambda x: 'Ensembl' if x.startswith('ENS') else 'UniProt')
    #
    isoforms.to_sql('isoforms', conn, if_exists='append', index=False)

def construct_transcript_table(conn, mapper):
    tables = get_tables(conn)

    #check to make sure gene and isoform tables are created
    if 'genes' not in tables:
        raise ValueError("Gene table is not created but is needed for transcript table. Run `construct_gene_tables` first.")
    if 'isoforms' not in tables:
        raise ValueError("Isoform table is not created but is needed for transcript table. Run `construct_isoform_tables` first.")
    
    #remove transcript table if it exists and talbe sthat reference transcript table
    if 'exons' in tables:
        conn.execute("DROP TABLE exons")
    if 'transcript_exon' in tables:
        conn.execute("DROP TABLE transcript_exon")
    if 'transcripts' in tables:
        conn.execute("DROP TABLE transcripts")
        

    create_table_query = f"""CREATE TABLE transcripts (
        Transcript_stable_ID TEXT PRIMARY KEY,
        Gene_stable_ID TEXT,
        APPRIS_annotation TEXT,
        Ensembl_Canonical TEXT,
        Transcript_support_level INT,
        TRIFID_Score FLOAT,
        Relative_CDS_Start INT,
        Relative_CDS_Stop INT,
        Isoform_ID TEXT,
        FOREIGN KEY (Isoform_ID) REFERENCES isoforms(Isoform_ID),
        FOREIGN KEY (Gene_stable_ID) REFERENCES genes(Gene_stable_ID)
    );"""
    #construct mapper table
    conn.execute(create_table_query)

    #construct dictionary connecting transcript ids to isoform ids in isoform table
    isoform_map = mapper.isoforms[['Transcript stable ID', 'Isoform ID']].drop_duplicates().copy()
    isoform_map['Transcript stable ID'] = isoform_map['Transcript stable ID'].str.split(';')
    isoform_map = isoform_map.explode('Transcript stable ID').reset_index(drop=True)

    #format transcript data into sql format
    columns_to_keep = ['Gene stable ID', 'APPRIS annotation', 'Ensembl Canonical', 'Transcript support level (TSL)', 'TRIFID Score', 'Relative CDS Start (bp)', 'Relative CDS Stop (bp)']
    transcripts = mapper.transcripts[columns_to_keep].reset_index().drop_duplicates().copy()
    #add isoform id
    transcripts = transcripts.merge(isoform_map, on='Transcript stable ID', how='left')

    transcripts = transcripts.dropna(subset = ['Relative CDS Start (bp)'])

    transcripts = transcripts.rename(columns={'Gene stable ID': 'Gene_stable_ID',
                            'APPRIS annotation': 'APPRIS_annotation',
                            'Ensembl Canonical': 'Ensembl_Canonical',
                            'Transcript support level (TSL)': 'Transcript_support_level',
                            'TRIFID Score': 'TRIFID_Score',
                            'Relative CDS Start (bp)': 'Relative_CDS_Start',
                            'Relative CDS Stop (bp)': 'Relative_CDS_Stop','Transcript stable ID': 'Transcript_stable_ID', 'Isoform ID': 'Isoform_ID'})

    transcripts.to_sql('transcripts', conn, if_exists='append', index = False)
    return transcripts


def construct_transcript_and_exons_table(conn, mapper):
    tables = get_tables(conn)

    #make sure gene and transcript tables are already created
    if 'genes' not in tables:
        raise ValueError("Gene table is not created but is needed for exon table. Run `construct_gene_tables` first.")
    if 'transcripts' in tables:
        conn.execute("""DROP TABLE transcripts""")
    if 'exons' in tables:
        conn.execute("""DROP TABLE exons""")

    transcripts = construct_transcript_table(conn, mapper)

    create_table_query = f"""CREATE TABLE exons (
        Exon_stable_ID TEXT PRIMARY KEY,
        Gene_stable_ID TEXT,
        Gene_Start INT,
        Gene_End INT,
        Exon_Length INT,
        FOREIGN KEY (Gene_stable_ID) REFERENCES genes(Gene_stable_ID)
    );"""
    #construct sql table
    conn.execute(create_table_query)

    #create exon table that contains exon-specific information
    exons = mapper.exons[['Exon stable ID', 'Gene stable ID', 'Exon Start (Gene)', 'Exon End (Gene)','Exon Length']].drop_duplicates()
    exons = exons.set_index('Exon stable ID')
    exons = exons.rename(columns={'Gene stable ID': 'Gene_stable_ID', 
                            'Exon Start (Gene)': 'Gene_Start', 
                            'Exon End (Gene)': 'Gene_End', 
                            'Exon Length': 'Exon_Length'})
    exons.to_sql('exons', conn, if_exists='append', index=True, index_label='Exon_stable_ID')


    ##### transcript_exon table #####
    tables = [table[0] for table in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    if 'transcript_exon' in tables:
        conn.execute("""DROP TABLE transcript_exon""")
    create_table_query = """CREATE TABLE transcript_exon (
        Exon_stable_ID TEXT,
        Transcript_stable_ID TEXT,
        Constitutive_exon BOOLEAN,
        Exon_rank_in_transcript INT,
        Transcript_start INT,
        Transcript_end INT,
        Protein_start FLOAT,
        Protein_end FLOAT,
        PRIMARY KEY (Exon_stable_ID, Transcript_stable_ID),
        FOREIGN KEY (Exon_stable_ID) REFERENCES exons(Exon_stable_ID),
        FOREIGN KEY (Transcript_stable_ID) REFERENCES transcripts(Transcript_stable_ID)
    );"""
    #construct mapper table
    conn.execute(create_table_query)

    columns_to_keep = ['Transcript stable ID', 'Exon stable ID', 'Constitutive exon', 'Exon rank in transcript', 'Exon Start (Transcript)', 'Exon End (Transcript)', 'Exon Start (Protein)', 'Exon End (Protein)']
    transcript_exon = mapper.exons[columns_to_keep].drop_duplicates().copy()
    #restrict to those in the transcripts table
    transcript_exon = transcript_exon[transcript_exon['Transcript stable ID'].isin(transcripts['Transcript_stable_ID'])]
    transcript_exon = transcript_exon.rename(columns={'Transcript stable ID': 'Transcript_stable_ID', 
                            'Exon stable ID': 'Exon_stable_ID', 
                            'Constitutive exon': 'Constitutive_exon', 
                            'Exon rank in transcript': 'Exon_rank_in_transcript',
                            'Exon Start (Transcript)': 'Transcript_start',  
                            'Exon End (Transcript)': 'Transcript_end',
                            'Exon Start (Protein)': 'Protein_start',
                            'Exon End (Protein)': 'Protein_end'})
    transcript_exon.to_sql('transcript_exon', conn, if_exists='append', index=False)


def construct_known_ptm_table(conn, mapper):
    tables = get_tables(conn)
    #make sure gene and transcript tables are already created
    if 'proteins' not in tables:
        raise ValueError("Exon table is not created but is needed for PTM info table. Run `create_proteins_table` first.")
    
    if 'isoform_ptms' in tables:
        conn.execute("""DROP TABLE isoform_ptms""")
    if 'ptm_info_and_coordinates' in tables:
        conn.execute("""DROP TABLE ptm_info_and_coordinates""")
    if 'ptm_coordinates' in tables:
        conn.execute("""DROP TABLE ptm_coordinates""")
    if 'known_ptms' in tables:
        conn.execute("""DROP TABLE ptm_info""")
    
    create_table_query = """CREATE TABLE known_ptms (
        PTM_ID TEXT PRIMARY KEY,
        UniProtKB_ID TEXT,
        SwissProt_ID TEXT,
        Residue TEXT,
        Position INT,
        Modification TEXT,
        Modification_class TEXT,
        Sources TEXT,
        Distance_to_N_Boundary INT,
        Distance_to_C_Boundary INT,
        Tryptic_fragment TEXT,
        Flanking_sequence TEXT,
        inDomain TEXT,
        Genomic_coordinates TEXT,
        Ragged BOOLEAN,
        PTM_Conservation_Score FLOAT,
        FOREIGN KEY (UniProtKB_ID) REFERENCES proteins(UniProtKB_ID)
    );"""
    #construct sql table
    conn.execute(create_table_query)

    #create ptm_info table that contains ptm-specific information
    ptm_info = mapper.ptm_info[['Protein', 'Residue', 'PTM Location (AA)', 'Modification', 'Modification Class', 'Sources', 'Distance to N-terminal Splice Boundary (NC)', 'Distance to C-terminal Splice Boundary (NC)', 'Tryptic Fragment', 'Flanking Sequence', 'Domain Type', 'Genomic Coordinates', 'Ragged', 'PTM Conservation Score']].reset_index().drop_duplicates().copy()
    ptm_info['SwissProt_ID'] = ptm_info['Protein'].str.split('-').str[0]
    ptm_info = ptm_info.rename(columns={'Protein': 'UniProtKB_ID', 
                            'Residue': 'Residue', 
                            'PTM Location (AA)': 'Position', 
                            'Modification': 'Modification', 
                            'Modification Class': 'Modification_class', 
                            'Sources': 'Sources', 
                            'Distance to N-terminal Splice Boundary (NC)': 'Distance_to_N_Boundary', 
                            'Distance to C-terminal Splice Boundary (NC)': 'Distance_to_C_Boundary',
                            'Tryptic Fragment': 'Tryptic_fragment',
                            'Flanking Sequence': 'Flanking_sequence',
                            'Domain Type': 'inDomain',
                            'Genomic Coordinates': 'Genomic_coordinates',
                            'Ragged': 'Ragged',
                            'PTM Conservation Score': 'PTM_Conservation_Score',
                            'index': 'PTM_ID'})

    ptm_info.to_sql('known_ptms', conn, if_exists='append', index=False)

    #set an index on SwissProt_ID
    query = """CREATE INDEX idx_swissprot_id ON known_ptms (SwissProt_ID);"""
    conn.execute(query)

def construct_ptm_coordinate_tables(conn, mapper):
    tables = get_tables(conn)

    if 'ptm_coordinates' in tables:
        conn.execute("""DROP TABLE ptm_coordinates""")
    create_table_query = """CREATE TABLE ptm_coordinates (
        Coordinate_ID TEXT PRIMARY KEY,
        Chromosome TEXT,
        Strand INT,
        Gene_location_hg38 INT
    );"""

    conn.execute(create_table_query)

    ptm_coordinates = mapper.ptm_coordinates[['Chromosome/scaffold name', 'Strand', 'Gene Location (hg38)']].reset_index().drop_duplicates().copy()
    ptm_coordinates = ptm_coordinates.rename(columns={'Genomic Coordinates': 'Coordinate_ID', 
                            'Chromosome/scaffold name': 'Chromosome', 
                            'Strand': 'Strand', 
                            'Gene Location (hg38)': 'Gene_location_hg38'})

    ptm_coordinates.to_sql('ptm_coordinates', conn, if_exists='append', index=False)


    #### table to map coordinates to specific ptms
    create_table_query = """CREATE TABLE ptm_info_and_coordinates (
        PTM_ID TEXT,
        Coordinate_ID TEXT,
        PRIMARY KEY (PTM_ID, Coordinate_ID),
        FOREIGN KEY (PTM_ID) REFERENCES known_ptms(PTM_ID),
        FOREIGN KEY (Coordinate_ID) REFERENCES ptm_coordinates(Coordinate_ID)
    );"""

    conn.execute(create_table_query)

    ptm_coordinates = mapper.ptm_coordinates.copy()
    ptm_coordinates = ptm_coordinates[['Source of PTM']].reset_index()
    ptm_coordinates = ptm_coordinates.rename(columns={'Source of PTM': 'PTM_ID', 'Genomic Coordinates': 'Coordinate_ID'})

    #split the 'Source of PTM' column into multiple rows if it contains multiple values
    ptm_coordinates['PTM_ID'] = ptm_coordinates['PTM_ID'].str.split(';')
    ptm_coordinates = ptm_coordinates.explode('PTM_ID').reset_index(drop=True)

    ptm_coordinates.to_sql('ptm_info_and_coordinates', conn, if_exists='append', index=False)

def construct_isoform_ptm_table(conn, mapper):
    #make isoform transcript table
    tables = get_tables(conn)

    if 'isoform_ptms' in tables:
        conn.execute("""DROP TABLE isoform_ptms""")
    create_table_query = """CREATE TABLE isoform_ptms (
        Isoform_ID TEXT,
        PTM_ID TEXT,
        Residue TEXT,
        Position INTEGER,
        Flanking_Sequence TEXT,
        Tryptic_Fragment TEXT,
        Conserved_Flank TEXT,
        FOREIGN KEY (Isoform_ID) REFERENCES isoforms (Isoform_ID),
        FOREIGN KEY (PTM_ID) REFERENCES known_ptms (PTM_ID)
    );"""

    conn.execute(create_table_query)

    mapper.getIsoformSpecificPTMs()

    isoform_ptms = mapper.isoform_ptms[mapper.isoform_ptms['Mapping Result'] == 'Success']

    cols_to_keep = ['Isoform ID', 'Source of PTM', 'Alternative Residue', 'Alternative Protein Position (AA)', 'Conserved Flank (Size = 5)', 'Flanking Sequence', 'Tryptic Fragment']
    isoform_ptms = isoform_ptms[cols_to_keep]

    #separate source of PTM into unique entries
    isoform_ptms['Source of PTM'] = isoform_ptms['Source of PTM'].str.split(';')
    isoform_ptms = isoform_ptms.explode('Source of PTM')

    #rename columns
    isoform_ptms.rename(columns={
        'Isoform ID': 'Isoform_ID',
        'Source of PTM': 'PTM_ID',
        'Alternative Residue': 'Residue',
        'Alternative Protein Position (AA)': 'Position',
        'Conserved Flank (Size = 5)': 'Conserved_Flank',
        'Flanking Sequence': 'Flanking_Sequence',
        'Tryptic Fragment': 'Tryptic_Fragment'
    }, inplace=True)


    isoform_ptms.to_sql('isoform_ptms', con=conn, if_exists='append', index=False)

def construct_all_tables(db_file, restart = False):
    if restart and os.path.exists(db_file):
        #delete existing database
        os.remove(db_file)
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON;")


    tables = get_tables(conn)
    if 'genes' not in tables:
        construct_gene_table(conn, mapper)
    if 'proteins' not in tables:
        construct_protein_tables(conn, mapper)
    if 'domains' not in tables:
        construct_domains_table(conn, mapper)
    if 'isoforms' not in tables:
        construct_isoform_tables(conn, mapper)
    #if 'transcripts' not in tables:
    #    construct_transcript_table(conn, mapper)
    if 'exons' not in tables:
        construct_transcript_and_exons_table(conn, mapper)
    if 'known_ptms' not in tables:
        construct_known_ptm_table(conn, mapper)
    if 'ptm_coordinates' not in tables:
        construct_ptm_coordinate_tables(conn, mapper)
    if 'isoform_ptms' not in tables:
        construct_isoform_ptm_table(conn, mapper)


def main():
    db_file = 'mapper.db'
    construct_all_tables(db_file, restart=True)
    print("Database construction complete.")

if __name__ == "__main__":
    main()