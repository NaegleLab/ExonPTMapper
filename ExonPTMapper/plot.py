import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
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
               from dataframe to initialize class variables instead (dataframe should be generated previously using self class)
               
        Returns
        -------
        self class
        """
        #load necessary data
        self.load_plotter()
        
    def plotTranscripts(self, gene_id = None, gene_name = None, transcript_id = None, functional_threshold = 0, sort_by_function = True, coding_color = 'red', noncoding_color = 'white'):
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
            gene_id = self.transcripts.loc[transcript_id, 'Gene stable ID']
            gene_exons = self.exons[self.exons['Gene stable ID'] == gene_id]
        else:
            raise ValueError('No gene or transcript indicate. Please provide gene_id, gene_name, or transcript_id')
        
        
        #get relevant transcripts
        transcript_ids = gene_exons['Transcript stable ID'].unique()
        tmp_trans = self.transcripts.loc[transcript_ids]
        #remove transcripts with missing coding information
        tmp_trans['Relative CDS Start (bp)'] = pd.to_numeric(tmp_trans['Relative CDS Start (bp)'], errors = 'coerce')
        missing_transcripts = list(tmp_trans.loc[tmp_trans['Relative CDS Start (bp)'].isna()].index.values)
        if len(missing_transcripts):
            print(f'Transcripts with Missing/Conflicting Coding Information: {",".join(missing_transcripts)}')
            
        tmp_trans = tmp_trans.dropna(subset = 'Relative CDS Start (bp)')
        
        #restrict to functional transcripts
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
            
            #get location of cds start and cds end
            tid_exons = gene_exons[gene_exons['Transcript stable ID'] == tid]
            gene_cds_start, gene_cds_end = getGenomicCodingRegion(self.transcripts.loc[tid], tid_exons, self.genes.loc[gene_id,'Strand']) 
            ax.annotate(tid, (gene_start-1500, num_transcripts - row +0.6), ha = 'right', va = 'center')
            ax.annotate(trans_type, (gene_start-1500, num_transcripts - row +0.4), ha = 'right', va = 'center')
            ax.plot([gene_start, gene_end], [num_transcripts-row+0.5, num_transcripts-row+0.5], c = 'k')
            for i in tid_exons.index:
                fiveprime = int(tid_exons.loc[i, "Exon Start (Gene)"])
                threeprime = int(tid_exons.loc[i, "Exon End (Gene)"])
                #check if exon is fully noncoding, fully coding, or if cds start/stop exists in exon. Plot exon accordingly
                if threeprime < gene_cds_start or fiveprime > gene_cds_end:
                    rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = noncoding_color, edgecolor = 'black', zorder = 2)
                    ax.add_patch(rect)
                elif fiveprime >= gene_cds_start and threeprime <= gene_cds_end:
                    rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = coding_color, edgecolor = 'black', zorder = 2)
                    ax.add_patch(rect)
                else:
                    noncoding_rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = noncoding_color, edgecolor = 'black', zorder = 2)
                    ax.add_patch(noncoding_rect)
                    
                    if fiveprime < gene_cds_start:
                        fiveprime = gene_cds_start
                    if threeprime > gene_cds_end:
                        threeprime = gene_cds_end
                    
                    noncoding_rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = coding_color, edgecolor = 'black', zorder = 3)
                    ax.add_patch(noncoding_rect)
                    

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
        ptms = self.ptm_info[self.ptm_info['Protein'] == protein_id]
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
        
        
    def plotSequenceAroundPTM(self, ptm, transcript_id = None, transcript_type = 'Canonical', orientation = 'top', aa_size = 20, 
                 transcript_color = 'gray', protein_color = 'lightblue', ptm_color = 'red', return_coordinates = False, ax = None):
        """
        Overlay protein sequence and transcript sequence surrounding a post translation so that codon in transcript matches corresponding residue and indicates whether residue is modified. Best for showing small sections of a protein.
        
        Parameters
        ----------
        ptm: str
            Indicates the ptm of interest (Ex. P32004_S1181)
        transcript_id: str
            Ensembl transcript ID to plot. If transcript_type is alternative, this is required.
        transcript_type: str
            Either 'Canonical' or 'Alternative' and indicates whether PTM being plotted is for a canonical or alternative protein
        orientation: str
            Indicates where protein coordinates should be plotted. 'top' indicates it will be on top of plot, if 'bottom' will be on bottom.
        aa_size: int
            Number of residues to plot around the ptm of interest
        transcript_color: str
            color of bars to use for transcript sequence
        protein_color: str
            color of bars to use for protein sequence
        ptm_color: str
            color of text to use in protein sequence if residue is modified
        return_coordinates: boolean
            whether to return the transcript coordinates used for the plot
        ax: matplotlib axes object
            where to plot the figure
        """
        if transcript_type == 'Canonical':
            ptm_loc_col = 'PTM Location (AA)'
            protein = self.ptm_info[self.ptm_info['Protein'] == ptm.split('_')[0]]
            ptm = self.ptm_info.loc[ptm]
            ptm_loc = int(ptm[ptm_loc_col])
            if transcript_id is None:
                transcript_id = ptm['Transcripts']
                if ',' in transcript_id:
                    transcript_id = transcript_id.split(',')[0]
        elif transcript_type == 'Alternative':
            if transcript_id is None:
                raise ValueError('Transcript ID must be provided to plot alternative transcripts')

            ptm_loc_col = 'Alternative Protein Location (AA)'
            ptm = self.alternative_ptms[(self.alternative_ptms['Alternative Transcript'] == transcript_id) & (self.alternative_ptms['PTM'] == ptm)].squeeze()

            if ptm[ptm_loc_col] == ptm[ptm_loc_col]:
                ptm_loc = int(ptm[ptm_loc_col])
                protein = self.alternative_ptms[self.alternative_ptms['Alternative Transcript'] == transcript_id]
                protein = protein[protein['Mapping Result'] == 'Success']
            else:
                ptm_loc = int(ptm['Canonical Protein Location (AA)'])
                protein = self.alternative_ptms[self.alternative_ptms['Alternative Transcript'] == transcript_id]
                protein = protein[protein['Mapping Result'] == 'Success']
        #get aa and nc sequence
        aa_sequence = self.transcripts.loc[transcript_id, 'Amino Acid Sequence']
        nc_sequence = self.transcripts.loc[transcript_id, 'Transcript Sequence']
        cds_start = int(self.transcripts.loc[transcript_id, 'Relative CDS Start (bp)'])
        
        #calculate spread parameters
        start = int(ptm_loc-aa_size/2)
        stop = int(ptm_loc+aa_size/2)
        nc_start = (start -1)*3 -3 +cds_start
        nc_stop = (stop - 1)*3 -3 + cds_start
        
        
        #get map
        aa_to_nc_map = constructProtToRNAmap(len(aa_sequence), cds_start)
        
        #find relevant splice boundaries
        boundaries = self.transcripts.loc[transcript_id, 'Exon cuts'].split(',')
        boundaries = [int(b) for b in boundaries if int(b) >= nc_start and int(b) <= nc_stop]
        
        #construct figure
        if ax is None:
            fig, ax = plt.subplots(figsize = (10, 1))
        if orientation == 'top':
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            protein_height = 1.3
            transcript_height = 0.7
        else:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            protein_height = 0.7
            transcript_height = 1.3
            
        ax.set_xlim([start-1,stop+1])
        ax.set_ylim([0,2])

        for i in range(start, stop):
            pos = i
            if pos in protein[ptm_loc_col].values:
                c = ptm_color
            else:
                c = 'black'
            #add residue
            ax.text(i,protein_height,aa_sequence[i-1], c= c, va = 'center', ha = 'center')

            #add box around nc sequence


            #add nc sequence
            codon_loc = aa_to_nc_map[i-1]
            for j in range(3):
                plot_loc = i - 1/3 + j/3
                nc_loc = codon_loc[j]
                ax.text(plot_loc,transcript_height, nc_sequence[nc_loc], c = c, va = 'center', ha = 'center')
                #check if splice boundary exists at this nc, if so indicate on plot
                if nc_loc in boundaries:
                    if j != 0:
                        linestyle = 'dashed'
                    else:
                        linestyle = 'solid'
                    ax.axvline(plot_loc-1/6, linestyle=linestyle, linewidth = 1, c = 'black', ymin = 0.25, ymax = 0.8)

        ax.text(start-1, protein_height, f'{transcript_type} Protein', va = 'center', ha = 'right', fontsize = 11)
        ax.text(start-1, transcript_height, f'{transcript_type} Transcript', va = 'center', ha = 'right', fontsize = 11)
        ax.add_patch(patches.Rectangle((start-1/2,protein_height-0.2), stop-start, 0.5, alpha = 0.2, facecolor = protein_color))
        ax.add_patch(patches.Rectangle((start-1/2,transcript_height-0.2), stop-start, 0.5, alpha = 0.2, facecolor = transcript_color))
        #ax.add_patch(patches.Rectangle((start-1/2,0.5), stop-start, 1.05, alpha = 1, edgecolor = 'black', facecolor = None))
        #ax[row].text(i,0.5,seq2[i])

        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(f'Protein Coordinates ({start}-{stop})')
        
        if return_coordinates:
            nc_start = (start -1)*3 -3 +cds_start
            nc_stop = (stop - 1)*3 -3 + cds_start
            return nc_start, nc_stop
    

    def plotTranscriptRegion(self, transcript_id, start, stop, ax, coordinates = 'Transcript', plot_full_exons = False,
                       return_gene_coordinates = False, height = 1, invert = True):
        gene_id = self.transcripts.loc[transcript_id, 'Gene stable ID']
        strand = self.genes.loc[gene_id,'Strand']
        if coordinates == 'Transcript':
            exons = self.exons[self.exons['Transcript stable ID'] == transcript_id]
            case1 = (exons['Exon End (Transcript)'] > stop) & (exons['Exon Start (Transcript)'] < stop)
            case2 = (exons['Exon Start (Transcript)'] < start) & (exons['Exon End (Transcript)'] > start)
            case3 = (exons['Exon Start (Transcript)'] > start) & (exons['Exon End (Transcript)'] < stop)
            exons = exons[case1 | case2 | case3]
            if not plot_full_exons:
                if strand == 1:
                    start_buffer = (start - exons['Exon Start (Transcript)'].min())
                    stop_buffer = (exons['Exon End (Transcript)'].max() - stop)
                    gene_start = exons['Exon Start (Gene)'].min() + start_buffer
                    gene_end = exons['Exon End (Gene)'].max() - stop_buffer
                    gene_length = gene_end - gene_start
                else:
                    start_buffer = (exons['Exon End (Transcript)'].max() - stop)
                    stop_buffer = (start - exons['Exon Start (Transcript)'].min())
                    gene_start = exons['Exon Start (Gene)'].min() + start_buffer
                    gene_end = exons['Exon End (Gene)'].max() - stop_buffer
                    gene_length = gene_end - gene_start
            else:
                if strand == 1:
                    start_buffer = 0
                    stop_buffer = 0
                    gene_start = exons['Exon Start (Gene)'].min() + start_buffer
                    gene_end = exons['Exon End (Gene)'].max() - stop_buffer
                    gene_length = gene_end - gene_start
                else:
                    start_buffer = 0
                    stop_buffer = 0
                    gene_start = exons['Exon Start (Gene)'].min() + start_buffer
                    gene_end = exons['Exon End (Gene)'].max() - stop_buffer
                    gene_length = gene_end - gene_start
                    
        elif coordinates == 'Gene':
            exons = self.exons[self.exons['Transcript stable ID'] == transcript_id]
            case1 = (exons['Exon End (Gene)'] >= stop) & (exons['Exon Start (Gene)'] <= stop)
            case2 = (exons['Exon Start (Gene)'] <= start) & (exons['Exon End (Gene)'] >= start)
            case3 = (exons['Exon Start (Gene)'] >= start) & (exons['Exon End (Gene)'] <= stop)
            gene_start = start
            gene_end = stop
            gene_length = gene_end - gene_start
            exons = exons[case1 | case2 | case3]
            if plot_full_exons:
                start_buffer = 0
                stop_buffer = 0
            else:
                if strand == 1:
                    start_buffer = (gene_start - exons['Exon Start (Gene)'].min())
                    stop_buffer = (exons['Exon End (Gene)'].max() - gene_end)
                else:
                    start_buffer = (exons['Exon End (Gene)'].max() - gene_end)
                    stop_buffer = (gene_start - exons['Exon Start (Gene)'].min())

            
        #get CDS Start/Stop and map to location in gene if within exon range
        gene_cds_start, gene_cds_end = getGenomicCodingRegion(self.transcripts.loc[transcript_id], self.exons[self.exons['Transcript stable ID'] == transcript_id], strand)
        
            
        ax.plot([gene_start, gene_end], [height, height], c = 'k')
        #add dotted lines to signify continuing of gene
        ax.plot([gene_start-gene_length/10, gene_start], [height, height], c = 'k', linestyle = 'dashed')
        ax.plot([gene_end, gene_end + gene_length/10], [height, height], c = 'k', linestyle = 'dashed')
        for i in exons.index:
            exon_seq = exons.loc[i, 'Exon Sequence']
            if exons.loc[i, 'Exon Start (Gene)'] < gene_start:
                fiveprime = gene_start
                threeprime = int(exons.loc[i, "Exon End (Gene)"])
                exon_seq = exon_seq[start_buffer:]
            elif exons.loc[i, 'Exon End (Gene)'] > gene_end:
                fiveprime = int(exons.loc[i,'Exon Start (Gene)']) 
                threeprime = gene_end
                exon_seq = exon_seq[0:len(exon_seq)-stop_buffer]
            else:
                fiveprime = int(exons.loc[i, "Exon Start (Gene)"])
                threeprime = int(exons.loc[i, "Exon End (Gene)"])
            
            #add exon patches
                
            rect = patches.Rectangle((fiveprime,height-0.3), threeprime - fiveprime, 0.6, 
                                     zorder = 2, facecolor = 'white', edgecolor = 'black')
            ax.add_patch(rect)
            
            #add coding patches
            #check if complete exon is in coding region
            if fiveprime >= gene_cds_start and threeprime <= gene_cds_end:
                add_coding = True
            else: 
                if (fiveprime < gene_cds_start and threeprime > cds_start):
                    fiveprime = gene_cds_start
                    add_coding = True
                
                
                if fiveprime < gene_cds_end and threeprime > gene_cds_end:
                    threeprime = gene_cds_end
                    add_coding = True

            if add_coding:
                rect = patches.Rectangle((fiveprime,height-0.3), threeprime - fiveprime, 0.6, 
                             zorder = 2, facecolor = 'gray', edgecolor = 'black')
                ax.add_patch(rect)
        
        if strand == -1 and invert:
            ax.invert_xaxis()
            
        if return_gene_coordinates:
            return gene_start, gene_end

    def plotSpliceEvent(self, canonical_transcript_id, alternative_transcript_id, start, stop, plot_full_exons = False, ax = None):
        num_transcripts = 2
        if ax is None:
            fig, ax = plt.subplots(figsize = (15, 2))
            
        ax.set_ylim([0, 2])
        
        gene_start, gene_end = self.plotTranscriptRegion(canonical_transcript_id, start, stop, ax, plot_full_exons = plot_full_exons, return_gene_coordinates = True, height = 2-1+0.5)
        self.plotTranscriptRegion(alternative_transcript_id, gene_start, gene_end, ax, coordinates = 'Gene', 
                       plot_full_exons = True, height = 2-2+0.5, invert = False)
        ax.axis('off')
        return gene_start, gene_end

    def plotPTMChanges(self, ptm, alternative_transcript_id, aa_size = 26, plot_full_exons = False):
        transcript_id = self.ptm_info.loc[ptm, 'Transcripts']
        if ',' in transcript_id:
            transcript_id = transcript_id.split(',')[0]

        #setup subplot
        fig, axes = plt.subplots(nrows = 3, figsize = (10,4))
        fig.subplots_adjust(hspace = 0.5)
            
        nc_start, nc_stop = self.plotSequenceAroundPTM(ptm,transcript_id = transcript_id, transcript_type = 'Canonical', orientation = 'top', aa_size = aa_size, return_coordinates = True, ax = axes[0])
        gene_start, gene_end = self.plotSpliceEvent(transcript_id, alternative_transcript_id, nc_start, nc_stop, plot_full_exons = plot_full_exons, ax = axes[1])
        self.plotSequenceAroundPTM(ptm,transcript_id = alternative_transcript_id, transcript_type = 'Alternative', orientation = 'bottom', aa_size = aa_size, ax = axes[2])
        return gene_start, gene_end, transcript_id
        
        
        
    #def plotPTMs(self, transcript_id):
    #    """
    #    Currently broken, but idea is that transcript is plotted with location of ptms indicated, highlighting conserved vs. spliced
    #    ptms
    #    """
    #    fig, ax = plt.subplots(figsize = (15,2))

    #    ax.set_ylim([0,2])
    #    transcript = self.transcripts.loc[transcript_id]
    #    ptms = self.ptms[self.ptms['Transcript stable ID'] == transcript_id]
    #    if ptms.shape[0] > 0:
    #        #transcript box
    #        transcript_length = len(transcript['seq'].values[0])
    #        ax.set_xlim([-1,transcript_length+10])

            #add coding sequence

    #        cds_start = transcript["CDS Start"]
    #        cds_end = transcript["CDS Stop"].values[0]
    #        rect = patches.Rectangle((cds_start, 0.01), cds_end - cds_start, 1, color = 'red')
    #        ax.add_patch(rect)

            #get exon cuts
    #        cuts = transcript['Exon cuts'].values[0].split('[')[1].split(']')[0].split(',')
    #        start = 0
    #        for cut in cuts:
    #            cut = int(cut)
    #            rect = patches.Rectangle((start, 0.01), cut-start, 1, edgecolor = 'k', facecolor = None, fill = False)
    #            ax.add_patch(rect)
    #            start = cut

            #get ptm location
    #        ptm_colors = {'N-Glycosylation': 'pink', 'Phosphoserine': 'orange', 'Phosphothreonine':'green', 'Phosphotyrosine': 'brown',
    #                     'Ubiquitination': 'blue'}
    #        heights = [1.3, 1.5, 1.7]
    #        spot = 0
    #        direction = 'Forward'
    #        for i in ptms.index:
    #            h = heights[spot]
    #            tmp_ptm = ptms.loc[i]
    #            pos_aa = tmp_ptm['position']
    #            pos_nc = [pos_aa*3 - 3+cds_start, pos_aa*3+cds_start]
    #            mod = tmp_ptm['aa']
    #            mod_type = tmp_ptm['mod']
    #            rect = patches.Rectangle((pos_nc[0],1.03),3, h-1, color=ptm_colors[mod_type], edgecolor = None)
    #            ax.add_patch(rect)
    #            ax.annotate(mod, (pos_nc[0]+3, h), ha = 'center', va = 'center', fontsize = 9)
    #            circ = patches.Ellipse((pos_nc[0]+5, h+0.02), width = 150, height = 0.25, fill = False, edgecolor = ptm_colors[mod_type])
    #            ax.add_patch(circ)
    #            if direction == 'Forward':
    #                spot = spot + 1
    #                if spot == 2:
    #                    direction = 'Reverse'
    #            else:
    #                spot = spot - 1
    #                if spot == 0:
    #                    direction = 'Forward'
    #        #ax.plot([pos_nc[0],pos_nc[0]], [1,1.5], c = 'k')
    #        #ax.plot([pos_nc[1],pos_nc[1]], [1,1.5], c = 'k')

            #def addMod(loc, mod_type):
                
    #        ax.axis('off')
            
    def load_plotter(self):
        #check if each data file exists: if it does, load into self object
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
            
def explode_PTMinfo(ptm_info, explode_cols = ['Transcripts', 'Gene Location (NC)', 'Transcript Location (NC)', 'Exon Location (NC)', 'Exon stable ID', 'Exon Rank']):
    exploded_ptms = ptm_info.copy()
    for col in explode_cols:
        if col in exploded_ptms.columns:
            exploded_ptms[col] = exploded_ptms[col].apply(lambda x: x.split(','))
        else:
            exploded_cols = exploded_cols.remove(col)
  
    exploded_ptms = exploded_ptms.explode(explode_cols)
    return exploded_ptms
    
    
def constructProtToRNAmap(protein_length, cds_start):
    prot_to_rna_map = []
    for aa_loc in range(protein_length):
        first_codon_loc = cds_start + aa_loc*3
        codon_loc = list(range(first_codon_loc,first_codon_loc+3))
        prot_to_rna_map.append(codon_loc)
    return prot_to_rna_map
    
def getGenomicCodingRegion(transcript, exons, strand):
    #get CDS Start/Stop and map to location in gene if within exon range
    cds_start = int(transcript['Relative CDS Start (bp)'])
    cds_stop = int(transcript['Relative CDS Stop (bp)'])
    coding_start_exon = exons[(exons['Exon Start (Transcript)'] <= cds_start) & (exons['Exon End (Transcript)'] >= cds_start)].squeeze()
    coding_stop_exon = exons[(exons['Exon Start (Transcript)'] <= cds_stop) & (exons['Exon End (Transcript)'] >= cds_stop)].squeeze()
    if strand == 1:
        gene_cds_start = coding_start_exon['Exon Start (Gene)'] + (cds_start - coding_start_exon['Exon Start (Transcript)'])
        gene_cds_end = coding_stop_exon['Exon Start (Gene)'] + (cds_stop - coding_stop_exon['Exon Start (Transcript)'])
    else:
        gene_cds_end = coding_start_exon['Exon End (Gene)'] - (cds_start - coding_start_exon['Exon Start (Transcript)'])
        gene_cds_start = coding_stop_exon['Exon End (Gene)'] - (cds_stop - coding_stop_exon['Exon Start (Transcript)'])
        
    return gene_cds_start, gene_cds_end