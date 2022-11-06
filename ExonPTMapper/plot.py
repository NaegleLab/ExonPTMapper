import matplotlib.pyplot
import pandas as pd
import numpy as np


class plot:
    def __init__(self, exons, transcripts, ptms, genes):
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
        self.exons.index = exons['Exon stable ID']
        self.transcripts = transcripts
        self.ptms = ptms
        self.genes = genes
        #identify available transcripts for PTM analysis (i.e. those that match in gencode and PS)
        
        def plotTranscripts(self, gene_id = None, transcript_id = None):
            """
            Given a gene ID, plot all transcripts associated with that same gene. Given transcript ID, plot all transcripts associated with the same gene as the given transcript. Default to given gene ID.
            """
            #get gene specific transcripts
            if gene_id is not None:
                gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
            elif transcript_id is not None:
                gene_id = self.exons.loc[self.exons['Transcript stable ID'] == transcript_id, 'Gene stable ID'].unique()[0]
                gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
                
            #establish plot range   
            num_transcripts = gene_exons['Transcript stable ID'].nunique()
            fig, ax = plt.subplots(figsize = (15,num_transcripts))
            ax.set_ylim([0,num_transcripts])
            #transcript box
            gene_length = len(self.genes.loc[gene_id, 'Unspliced Gene'])
            ax.set_xlim([-2500,gene_length+2500])


            #for each transcript associated with gene, plot exons along gene axis
            row = 1
            for tid in gene_exons['Transcript stable ID'].unique():
                ax.annotate(tid, (-1500, num_transcripts - row +0.5), ha = 'right', va = 'center')
                tmp = self.exons[self.exons['Transcript stable ID'] == tid]
                ax.plot([0, gene_length], [num_transcripts-row+0.5, num_transcripts-row+0.5], c = 'k')
                for i in tmp.index:
                    fiveprime = int(tmp.loc[i, "Exon Start (Gene)"])
                    threeprime = int(tmp.loc[i, "Exon End (Gene)"])
                    rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = 'red', edgecolor = 'black')
                    ax.add_patch(rect)
                row = row + 1

            ax.axis('off')
            
        def plotPTMs(self, transcript_id):
            fig, ax = plt.subplots(figsize = (15,2))

            ax.set_ylim([0,2])
            transcript = self.transcripts.loc[transcript_id]
            ptms = self.ptms[self.ptms['Transcript'] == transcript_id]
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