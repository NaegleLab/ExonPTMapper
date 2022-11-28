import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from ExonPTMapper import config


class plotter:
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
        #load necessary data
        self.exons = pd.read_csv(config.processed_data_dir + 'exons.csv')
        self.transcripts = pd.read_csv(config.processed_data_dir + 'transcripts.csv', index_col = 0)
        ptm_info = pd.read_csv(config.processed_data_dir + 'ptm_info.csv', index_col = 0)
        self.ptms = explode_PTMinfo(ptm_info)
        self.genes = pd.read_csv(config.processed_data_dir + 'genes.csv', index_col = 0)
        #self.exons = exons
        #self.exons.index = exons['Exon stable ID']
        #self.transcripts = transcripts
        #self.ptms = ptms
        #self.genes = genes
        #identify available transcripts for PTM analysis (i.e. those that match in gencode and PS)
        
    def plotTranscripts(self, gene_id = None, gene_name = None, transcript_id = None, functional_threshold = 0, sort_by_function = True):
        """
        Given a gene ID, plot all transcripts associated with that same gene. Given transcript ID, plot all transcripts associated with the same gene as the given transcript. Default to given gene ID.
        """
        #get gene specific transcripts
        if gene_id is not None:
            gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
            if gene_exons.shape[0] == 0:
                raise ValueError('Gene ID not found in dataset')
        elif gene_name is not None:
            gene = self.genes[self.genes['Gene name'] == gene_name]
            if gene.shape[0] == 0:
                raise ValueError('Gene name not found in dataset')
            gene_id = gene.index.values[0]
            gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
            if gene_exons.shape[0] == 0:
                raise ValueError('No exons found associated with that gene')
        elif transcript_id is not None:
            gene_id = self.exons.loc[self.exons['Transcript stable ID'] == transcript_id, 'Gene stable ID'].unique()[0]
            gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
            if gene_exons.shape[0] == 0:
                raise ValueError('Transcript ID not found in dataset')
        else:
            raise ValueError('No gene or transcript indicate. Please provide gene_id, gene_name, or transcript_id')
        
        
        #get relevant transcripts
        transcript_ids = gene_exons['Transcript stable ID'].unique()
        tmp_trans = self.transcripts.loc[transcript_ids]
        tmp_trans = tmp_trans[tmp_trans['TRIFID Score'] >= functional_threshold]
        if sort_by_function:
            tmp_trans = tmp_trans.sort_values(by = 'TRIFID Score', ascending = False)
        transcript_ids = tmp_trans.index.values
        
        #establish plot range   
        num_transcripts = len(transcript_ids)
        fig, ax = plt.subplots(figsize = (15,num_transcripts))
        ax.set_ylim([0,num_transcripts])
        #transcript box
        gene_start = self.genes.loc[gene_id, 'Gene start (bp)']
        gene_end = self.genes.loc[gene_id, 'Gene end (bp)']
        ax.set_xlim([gene_start-2500,gene_end+2500])


        #for each transcript associated with gene, plot exons along gene axis
        row = 1
        for tid in transcript_ids:
            #check whether transcript is canonical or alternative uniprot isoform
            if config.translator.loc[config.translator['Transcript stable ID'] == tid, 'Uniprot Canonical'].values[0] == 'Canonical':
                trans_type = '(canonical)'
            else:
                trans_type = '(alternative)'
            
            ax.annotate(tid, (gene_start-1500, num_transcripts - row +0.6), ha = 'right', va = 'center')
            ax.annotate(trans_type, (gene_start-1500, num_transcripts - row +0.4), ha = 'right', va = 'center')
            tmp = gene_exons[gene_exons['Transcript stable ID'] == tid]
            ax.plot([gene_start, gene_end], [num_transcripts-row+0.5, num_transcripts-row+0.5], c = 'k')
            for i in tmp.index:
                fiveprime = int(tmp.loc[i, "Exon Start (Gene)"])
                threeprime = int(tmp.loc[i, "Exon End (Gene)"])
                rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = 'red', edgecolor = 'black')
                ax.add_patch(rect)
            row = row + 1
        ax.set_title(self.genes.loc[gene_id, 'Gene name'])
        ax.axis('off')
        
    def annotateProteinSequence(self, protein_id = None, gene_name = None,include_exons = True,include_domains = False, num_aa_per_row = 100, figwidth = 20):
        """
        Print an annotated version of the canonical amino acid sequence of a given gene/protein. Indicate which residues are PTMs and where different splice boundaries lie on the protein. If indicated, include exon ID information or location of domains in the protein sequence.
        """
        #given protein id, find exons
        if protein_id is not None:
            trans = self.proteins.loc[protein_id, 'Canonical Transcripts']
            if ',' in trans:
                trans = trans.split(',')[0]
                
        elif gene_name is not None:
            transcripts = self.genes.loc[self.genes['Gene name'] == gene_name, 'Protein coding transcripts']
            canonicals = config.translator[config.translator['Uniprot Canonical'] == 'Canonical']
            if transcripts.shape[0] > 0:
                transcripts = transcripts.values[0].split(',')
                for trans in transcripts:
                    protein_id = canonicals.loc[canonicals['Transcript stable ID'] == trans, 'UniProtKB/Swiss-Prot ID']
                    if protein_id.shape[0] !=0:
                        protein_id = protein_id.values[0]
                        break
            else:
                raise ValueError('Gene name not found')

        seq = self.transcripts.loc[trans, 'Amino Acid Sequence']
        exons = self.exons.loc[self.exons['Transcript stable ID'] == trans]
        keep_coding = (exons['Exon Start (Protein)'] != "5' NCR") &  (exons['Exon Start (Protein)'] != "3' NCR") & (exons['Exon Start (Protein)'] != "No coding seq")
        coding_exons = exons[keep_coding]
        aa_cuts = coding_exons['Exon Start (Protein)'].astype(float).values
        ptms = self.ptms[self.ptms['Protein'] == protein_id]
        numrows = math.ceil(len(seq)/num_aa_per_row)
        fig, ax = plt.subplots(nrows = numrows, figsize = (figwidth, (1+include_domains*0.5)*numrows))
        fig.subplots_adjust(hspace = 1)
        if include_domains:
            pfam = config.ps_api.get_domains(protein_id, 'pfam')
        for row in range(numrows):
            #get where in the protein sequence to start for this row
            start = row*num_aa_per_row
            stop = (row+1)*num_aa_per_row
            ax[row].set_xlim([start,stop])
            
            #determine the size of y-axis depending on number of attributes to include
            num_attributes = 1+include_exons*1+include_domains*1
            ax[row].set_ylim([0,num_attributes])
            #check for splice boundaries in the start/stop range
            splice_bool = np.array([cut < stop and cut >= start for cut in aa_cuts])
            boundaries = aa_cuts[splice_bool]
            for bound in boundaries:
                #color based on ragged or normal boundary
                if bound % 1 != 0:
                    c = 'red'
                else:
                    c = 'blue'
                ax[row].plot([bound,bound],[0,1], c= c)
            for i in range(start, stop):
                pos = i + 1
                if pos in ptms['PTM Location (AA)'].values:
                    c = 'red'
                else:
                    c = 'black'
                #add residue if failed
                try:
                    ax[row].text(i+1,0.5,seq[i], c= c)
                except:
                    break
    #            if seq1[i] != seq2[i]:
    #                ax[row].add_patch(patches.Rectangle((i-0.25,0), 1, 2, alpha = 0.2))

            if include_exons:
                plot_exons = coding_exons[((coding_exons['Exon Start (Protein)'].astype(float) >= start) & (coding_exons['Exon Start (Protein)'].astype(float) <= stop)) | ((coding_exons['Exon End (Protein)'].astype(float) <= stop) & (coding_exons['Exon End (Protein)'].astype(float) >= start))]
                for i in plot_exons.index:
                    exon_start = float(plot_exons.loc[i,'Exon Start (Protein)'])
                    exon_end = float(plot_exons.loc[i, 'Exon End (Protein)'])
                    if start > exon_start:
                        exon_start = start
                    if stop < exon_end:
                        exon_end = stop
                        
                    #check if exon is constituitive
                    if plot_exons.loc[i,'Constitutive exon'] == 1:
                        c = 'plum'
                    else:
                        c = 'lightsteelblue'
                    rect = patches.Rectangle((exon_start, 1.1), exon_end - exon_start, 0.8, edgecolor = 'black', linewidth = 1, facecolor = c)
                    ax[row].add_patch(rect)
                    ax[row].text(exon_end - (exon_end-exon_start)/2, 1.5, plot_exons.loc[i, 'Exon stable ID'], c= 'white',ha = 'center', va = 'center')
                ax[row].text(start-1, 1.5,'Exon stable ID', ha = 'right', va = 'center')
                
            if include_domains:
                height = 1.5+include_exons*1
                ax[row].axhline(height, zorder = 0, c= 'black')
                for domain in pfam:
                    if (int(domain[1]) < start) and int(domain[2]) > stop:
                        rect = patches.Rectangle((int(start), height-0.4), int(stop)-int(start), 0.8, facecolor = 'white', edgecolor = 'black',linewidth = 1, zorder = 1)
                        ax[row].add_patch(rect)
                        ax[row].text(int(stop)-(stop-start)/2, height, domain[0], ha = 'center', va = 'center')
                    if (int(domain[1]) > start-1 and int(domain[1]) < stop) and (int(domain[2]) > start and int(domain[2]) < stop):
                        rect = patches.Rectangle((int(domain[1]), height-0.4), int(domain[2])-int(domain[1]), 0.8, facecolor = 'white', edgecolor = 'black',linewidth = 1, zorder = 1)
                        ax[row].add_patch(rect)
                        ax[row].text(int(domain[2])-(int(domain[2])-int(domain[1]))/2, height, domain[0], ha = 'center', va = 'center')
                    elif (int(domain[1]) > start and int(domain[1]) < stop):
                        rect = patches.Rectangle((int(domain[1]), height-0.4), int(stop)-int(domain[1]), 0.8, facecolor = 'white', edgecolor = 'black',linewidth = 1, zorder = 1) 
                        ax[row].add_patch(rect)
                        ax[row].text(stop-(stop-int(domain[1]))/2, height, domain[0], ha = 'center', va = 'center')
                    elif (int(domain[2]) > start and int(domain[2]) < stop):
                        rect = patches.Rectangle((start, height-0.4), int(domain[2])-int(start), 0.8, facecolor = 'white', edgecolor = 'black',linewidth = 1, zorder = 1)
                        ax[row].add_patch(rect)
                        ax[row].text(int(domain[2])-(int(domain[2])-start)/2, height, domain[0], ha = 'center', va = 'center')
                ax[row].text(start-1, 1.5+include_domains*1,'Domains', ha = 'right', va = 'center')
                        
                
            ax[row].spines["top"].set_visible(False)
            ax[row].spines["right"].set_visible(False)
            ax[row].spines["left"].set_visible(False)
            ax[row].get_yaxis().set_visible(False)

    fig.suptitle(gene_name)
        
    def plotPTMs(self, transcript_id):
        """
        Currently broken, but idea is that transcript is plotted with location of ptms indicated, highlighting conserved vs. spliced
        ptms
        """
        fig, ax = plt.subplots(figsize = (15,2))

        ax.set_ylim([0,2])
        transcript = self.transcripts.loc[transcript_id]
        ptms = self.ptms[self.ptms['Transcript stable ID'] == transcript_id]
        if ptms.shape[0] > 0:
            #transcript box
            transcript_length = len(transcript['seq'].values[0])
            ax.set_xlim([-1,transcript_length+10])

            #add coding sequence

            cds_start = transcript["CDS Start"]
            cds_end = transcript["CDS Stop"].values[0]
            rect = patches.Rectangle((cds_start, 0.01), cds_end - cds_start, 1, color = 'red')
            ax.add_patch(rect)

            #get exon cuts
            cuts = transcript['Exon cuts'].values[0].split('[')[1].split(']')[0].split(',')
            start = 0
            for cut in cuts:
                cut = int(cut)
                rect = patches.Rectangle((start, 0.01), cut-start, 1, edgecolor = 'k', facecolor = None, fill = False)
                ax.add_patch(rect)
                start = cut

            #get ptm location
            ptm_colors = {'N-Glycosylation': 'pink', 'Phosphoserine': 'orange', 'Phosphothreonine':'green', 'Phosphotyrosine': 'brown',
                         'Ubiquitination': 'blue'}
            heights = [1.3, 1.5, 1.7]
            spot = 0
            direction = 'Forward'
            for i in ptms.index:
                h = heights[spot]
                tmp_ptm = ptms.loc[i]
                pos_aa = tmp_ptm['position']
                pos_nc = [pos_aa*3 - 3+cds_start, pos_aa*3+cds_start]
                mod = tmp_ptm['aa']
                mod_type = tmp_ptm['mod']
                rect = patches.Rectangle((pos_nc[0],1.03),3, h-1, color=ptm_colors[mod_type], edgecolor = None)
                ax.add_patch(rect)
                ax.annotate(mod, (pos_nc[0]+3, h), ha = 'center', va = 'center', fontsize = 9)
                circ = patches.Ellipse((pos_nc[0]+5, h+0.02), width = 150, height = 0.25, fill = False, edgecolor = ptm_colors[mod_type])
                ax.add_patch(circ)
                if direction == 'Forward':
                    spot = spot + 1
                    if spot == 2:
                        direction = 'Reverse'
                else:
                    spot = spot - 1
                    if spot == 0:
                        direction = 'Forward'
            #ax.plot([pos_nc[0],pos_nc[0]], [1,1.5], c = 'k')
            #ax.plot([pos_nc[1],pos_nc[1]], [1,1.5], c = 'k')

            #def addMod(loc, mod_type):
                
            ax.axis('off')
            
def explode_PTMinfo(ptm_info, explode_cols = ['Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon Rank']):
    exploded_ptms = ptm_info.copy()
    for col in explode_cols:
        if col in exploded_ptms.columns:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(','))
        else:
            exploded_cols = exploded_cols.remove(col)
  
    exploded_ptms = exploded_ptms.explode(explode_cols)
    return exploded_ptms