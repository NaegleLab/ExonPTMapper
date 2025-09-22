import sqlite3
import pandas as pd
import plotly.graph_objects as go

class mapper_db:
    def __init__(self, conn):
        self.conn = conn
        self.tables = self.get_table_names()


    def close(self):
        self.conn.close()

    def get_table_names(self):
        tables = [table[0] for table in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        return tables

    def get_table_columns(self, table_name):
        query = f"""PRAGMA table_info({table_name})"""
        columns = [col[1] for col in self.conn.execute(query).fetchall()]
        return columns  

    def get_gene_id(self, gene_name):
        query = """SELECT Gene_stable_ID FROM genes WHERE Gene_name = ?"""

        cursor = self.conn.execute(query, (gene_name,))
        gene_stable_id = cursor.fetchone()[0]
        return gene_stable_id

    
    def convert_to_swissprot(self, gene_name):
        gene_stable_id = self.get_gene_id(gene_name)

        query = """SELECT SwissProt_ID FROM gene_to_proteins WHERE Gene_stable_ID = ?"""
        cursor = self.conn.execute(query, (gene_stable_id,))
        swissprot_id = cursor.fetchone()[0]
        return swissprot_id

    def get_canonical_isoform(self, id, id_type = 'UniProt'):
        if id_type == 'UniProt':
            query = """SELECT UniProtKB_ID FROM proteins WHERE SwissProt_ID = ? and UniProt_Isoform_Type = 'Canonical'"""
        elif id_type == 'Name':
            query = """SELECT Isoform_ID FROM isoforms
            JOIN genes ON isoforms.Gene_stable_ID = genes.Gene_stable_ID
            WHERE genes.Gene_name = ? and isoforms.Isoform_Type = 'Canonical'"""
        
        id = self.conn.execute(query, (id,)).fetchone()[0]
        return id

    def get_isoform_ids(self, id, id_type = 'UniProt'):
        if id_type == 'UniProt':
            query = """
            SELECT isoforms.Isoform_ID
            FROM isoforms
            JOIN gene_to_proteins ON isoforms.Gene_stable_ID = gene_to_proteins.Gene_stable_ID
            WHERE gene_to_proteins.SwissProt_ID = ? ORDER BY isoforms.Isoform_Length DESC
            """
        elif id_type == 'Name':
            query = """
            SELECT isoforms.Isoform_ID
            FROM isoforms
            JOIN genes ON isoforms.Gene_stable_ID = genes.Gene_stable_ID
            WHERE genes.Gene_name = ? ORDER BY isoforms.Isoform_Length DESC
            """
        isoform_ids = self.conn.execute(query, (id,)).fetchall()
        #unpack
        isoform_ids = [row[0] for row in isoform_ids]
        return isoform_ids
    
    def get_transcript_ids(self, id, id_type = 'Isoform', sort_by_function = True, TRIFID_threshold = 0):
        if id_type == 'Isoform':
            query = """SELECT Transcript_stable_ID FROM transcripts WHERE Isoform_ID = ? AND TRIFID_score >= ?"""
        elif id_type == 'Gene Name':
            id = self.get_gene_id(id)
            query = """SELECT Transcript_stable_ID FROM transcripts WHERE Gene_stable_ID = ? AND TRIFID_score >= ?"""
        elif id_type == 'Gene ID':
            query = """SELECT Transcript_stable_ID FROM transcripts WHERE Gene_stable_ID = ? AND TRIFID_score >= ?"""
        else:
            raise ValueError("id_type must be 'Isoform'")
        
        if sort_by_function:
            query += """ ORDER BY TRIFID_score DESC"""

        transcript_ids = self.conn.execute(query, (id,TRIFID_threshold)).fetchall()
        #unpack
        transcript_ids = [row[0] for row in transcript_ids]
        return transcript_ids

    def get_transcript_info(self, id, id_type = 'Isoform'):
        if id_type == 'Isoform':
            query = """SELECT * FROM transcripts WHERE Isoform_ID = ?"""
        elif id_type == 'Gene Name':
            id = self.get_gene_id(id)
            query = """SELECT * FROM transcripts WHERE Gene_stable_ID = ?"""
        else:
            raise ValueError("id_type must be 'Isoform'")
        transcript_info = self.conn.execute(query, (id,)).fetchall()
        return transcript_info
    
    def check_transcript_type(self, id, id_type = 'Transcript'):
        if id_type == 'Transcript':
            query = """SELECT isoforms.Isoform_Type FROM isoforms 
                JOIN transcripts ON isoforms.Isoform_ID = transcripts.Isoform_ID
                WHERE transcripts.Transcript_stable_ID = ?"""
        elif id_type == 'Isoform':
            query = """SELECT Isoform_Type FROM isoforms WHERE Isoform_ID = ?"""
        else:
            raise ValueError("id_type must be 'Transcript' or 'Isoform'")
        transcript_type = mapper_db.conn.execute(query, (id,)).fetchone()[0]
        return transcript_type


    def get_exon_info(self, id, id_type = 'Isoform'):
        if id_type == 'Isoform':
           #get transcript associated with isoform
            query = """SELECT Transcript_stable_ID FROM transcripts WHERE Isoform_ID = ?"""
            transcript = self.conn.execute(query, (id,)).fetchone()[0]
        elif id_type == 'Transcript':
            transcript = id
        #get exons
        query = """SELECT * FROM transcript_exon WHERE Transcript_stable_ID = ?"""
        exons = self.conn.execute(query, (transcript,)).fetchall()
        return exons
    
    def get_domain_info(self, id, id_type = "UniProt"):
        if id_type == 'UniProt':
            query = """SELECT * FROM domains WHERE SwissProt_ID = ?"""
        elif id_type == 'Name':
            query = """SELECT * FROM domains 
                    JOIN gene_to_proteins ON domains.SwissProt_ID = gene_to_proteins.SwissProt_ID
                    JOIN genes ON gene_to_proteins.Gene_stable_ID = genes.Gene_stable_ID
                    WHERE genes.Gene_name = ?"""
        #elif id_type == 'Isoform':
        #    query = """SELECT * FROM domains 
        #            JOIN isoforms ON domains.SwissProt_ID = isoforms.SwissProt_ID
        #            WHERE isoforms.Isoform_ID = ?"""
        else:
            raise ValueError("id_type must be 'UniProt', 'Name'")

        domains = self.conn.execute(query, (id,)).fetchall()
        return domains

    def get_canonical_ptm_table(self, id, id_type = 'Name', include_constitutive = False):
        if id_type == 'Name':
            prot_id = self.convert_to_swissprot(id)
            query = f"""SELECT Residue, Position, Modification_Class, ptm_conservation_score 
                    FROM known_ptms WHERE SwissProt_ID = ?"""
        elif id_type == 'Isoform':
            prot_id = id
            query = f"""SELECT Residue, Position, Modification_Class, ptm_conservation_score 
                    FROM known_ptms WHERE UniProtKB_ID = ?"""
        else:
            prot_id = id
            query = f"""SELECT Residue, Position, Modification_Class, ptm_conservation_score 
                    FROM known_ptms WHERE SwissProt_ID = ?"""


        if not include_constitutive:
            query += " AND ptm_conservation_score < 1"

        query += " ORDER BY Position"
        results = self.conn.execute(query, (prot_id,)).fetchall()

        #construct and output table
        results = pd.DataFrame(results, columns=["Residue", "Position", "Modification_Class", "ptm_conservation_score"])
        return results
    
    def get_ptm_table(self, isoform_id, include_constitutive = False):
        query = """SELECT isoform_ptms.Residue, isoform_ptms.Position, known_ptms.Modification_Class, known_ptms.ptm_conservation_score, known_ptms.Position, isoform_ptms.Conserved_Flank FROM isoform_ptms
            JOIN known_ptms ON known_ptms.PTM_ID = isoform_ptms.PTM_ID
            WHERE Isoform_ID = ?"""
        ptms = self.conn.execute(query, (isoform_id,)).fetchall()


        if not include_constitutive:
            query += " AND known_ptms.ptm_conservation_score < 1"

        query += " ORDER BY isoform_ptms.Position"

        ptms = self.conn.execute(query, (isoform_id,)).fetchall()
        ptms = pd.DataFrame(ptms, columns=["Residue", "Position", "Modification_Class", "ptm_conservation_score", "Canonical_Position", 'Conserved_Flank'])
        return ptms




