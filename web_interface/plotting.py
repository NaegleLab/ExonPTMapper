from matplotlib import patches
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import plotly.graph_objects as go

import sys
sys.path.append('..')
import sql_interface

class plotter(sql_interface.mapper_db):
    def __init__(self, conn):
        super().__init__(conn)

    def plot_canonical_isoform(self, text_id, id_type = 'Name'):
        canonical_isoform = self.get_canonical_isoform(text_id, id_type = id_type)
        #make sure to remove canonical tag
        domains = self.get_domain_info(text_id, id_type = id_type)

        #plot isoform
        fig = self.plot_isoform(canonical_isoform, domains = domains)
        return fig

    
    def plot_isoform(self, isoform_id, domains = None):
        exons = self.get_exon_info(isoform_id, id_type = 'Isoform')
        ptms = self.get_ptm_table(isoform_id, include_constitutive = True)
        fig = plot_protein_info(exons, ptms, domains = domains)
        return fig
    
    def plot_transcripts(self, id, id_type = 'Gene Name', functional_threshold = 0, sort_by_function = True):
        fig = plot_transcripts(self, id, id_type = id_type, functional_threshold = functional_threshold, sort_by_function = sort_by_function)
        return fig
    
    def add_exon(fig, start, stop, fillcolor = 'red', trace_label = ''):
        #add spliced region
        fig.add_shape(
            type="rect",
            x0=start, x1=stop,
            y0=0.4, y1=0.5,
            line=dict(color="black"),
            fillcolor=fillcolor,
        )

        fig.add_trace(go.Scatter(
            x=[start, stop, stop, start, start],
            y=[0.4, 0.4, 0.5, 0.5, 0.4],
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color="black"),
            mode="lines",
            name=trace_label,
            showlegend=False
        ))
        return fig
    
    def get_ptm_info(self, chromosome, strand, region):
        if strand == '+' or strand == 1:
            strand2 = 1
        else:
            strand2 = -1
        chromosome = chromosome.replace('chr', '')
        query = """SELECT known_ptms.PTM_ID, known_ptms.Residue, known_ptms.Position, known_ptms.Modification_Class, known_ptms.PTM_Conservation_Score, ptm_coordinates.Gene_location_hg38 FROM known_ptms
                        JOIN ptm_info_and_coordinates
                        ON known_ptms.PTM_ID = ptm_info_and_coordinates.PTM_ID
                        JOIN ptm_coordinates
                        ON ptm_info_and_coordinates.Coordinate_ID = ptm_coordinates.Coordinate_ID
                WHERE ptm_coordinates.Chromosome = ?
                    AND ptm_coordinates.Strand = ?
                AND ptm_coordinates.Gene_location_hg38 BETWEEN ? AND ?;"""
        ptm_info = self.conn.execute(query, (chromosome, strand2, int(region[0]), int(region[1]))).fetchall()
        return ptm_info

    def add_ptms_to_gene(self, fig, chromosome, strand, region):
        #get ptms in spliced region
        ptms = self.get_ptm_info(chromosome, strand, region)
        for ptm in ptms:
            #color based on whether constitutive
            if ptm[4] == 1:
                color = 'gray'
            else:
                color = 'darksalmon'

            #check if canonical position is provided if so, include in trace
            trace = f"PTM Site: {ptm[1]}{ptm[2]}<br>Modification: {ptm[3]}<br>Fraction of Isoforms: {round(ptm[4], 2)}"

            fig = add_ptm(fig, ptm[5], trace_label = trace, color = color)
        return fig

    def plot_splice_event(self, rmats_event, event_type = 'SE'):
        spliced_region, upstream_region, downstream_region, chromosome, strand = extract_event_info(rmats_event, event_type = event_type)

        fig = go.Figure()
        #add intron line
        fig.add_shape(
            type="rect",
            x0=upstream_region[0], x1=downstream_region[1],
            y0=0.445, y1=0.455,
            line=dict(color="black"),
            fillcolor='black',
        )
        #add spliced region
        fig = add_exon(fig, spliced_region[0], spliced_region[1], fillcolor='red', trace_label=f"Spliced Region<br>(dPSI={rmats_event['IncLevelDifference']:.2f})")
        fig = add_exon(fig, upstream_region[0], upstream_region[1], fillcolor='grey', trace_label="Upstream Region")
        fig = add_exon(fig, downstream_region[0], downstream_region[1], fillcolor='grey', trace_label="Downstream Region")


        #get ptms in spliced region
        fig = self.add_ptms_to_gene(fig, chromosome, strand, spliced_region)
        #get ptms in upstream region
        fig = self.add_ptms_to_gene(fig, chromosome, strand, upstream_region)
        #get ptms in downstream region
        fig = self.add_ptms_to_gene(fig, chromosome, strand, downstream_region)




        #remove x axis ticks, if on reverse strand reverse x axis
        if strand == -1:
            fig.update_xaxes(autorange="reversed", ticks="", showticklabels=False)
            annotation_start = downstream_region[1] + 100
        else:
            fig.update_xaxes(ticks="", showticklabels=False)
            annotation_start = upstream_region[0] - 100

        title = f"Splice Event: {rmats_event['geneSymbol']} | chr{chromosome}:{strand}:{spliced_region[0]}-{spliced_region[1]}"
        fig.update_layout(
            yaxis_visible=False,
            xaxis_visible=False,
            shapes=[],
            height=300,
            width=700,
            plot_bgcolor="rgba(0,0,0,0)",
            title = title
        )

        #add annotations indicating exon and ptm rows
        fig.add_annotation(
            x=annotation_start,
            y=0.51,
            text="PTMs",
            showarrow=False,
            font=dict(size=16),
            xanchor = 'right'
        )

        fig.add_annotation(
            x=annotation_start,
            y=0.45,
            text="Exons",
            showarrow=False,
            font=dict(size=16),
            xanchor = 'right'
        )


        return fig

def extract_event_info(rmats_event, event_type = 'SE'):
    if event_type == 'SE':
        spliced_region = (rmats_event['exonStart_0base'], rmats_event['exonEnd'])
        upstream_region = (rmats_event['upstreamES'], rmats_event['upstreamEE'])
        downstream_region = (rmats_event['downstreamES'], rmats_event['downstreamEE'])
        chromosome = rmats_event['chr'].replace('chr', '')
        strand_dict = {'+':1, '-':-1}
        strand = strand_dict[rmats_event['strand']]
    else:
        raise NotImplementedError(f"Event type {event_type} not implemented")

    return spliced_region, upstream_region, downstream_region, chromosome, strand

def get_transcript_info(mapper_db, id, id_type = 'Gene Name', functional_threshold = 0, sort_by_function = True):
    transcript_ids = mapper_db.get_transcript_ids(id, id_type = id_type)
    if len(transcript_ids) == 0:
        raise ValueError(f'No transcripts found for {id} of type {id_type}')



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

    if id_type == 'Gene Name':
        gene_id = mapper_db.get_gene_id(id)
    elif id_type == 'Gene ID':
        gene_id = id

    gene_start = mapper_db.genes.loc[gene_id, 'Gene start (bp)']
    gene_end = mapper_db.genes.loc[gene_id, 'Gene end (bp)']
    
    return transcript_ids, gene_start, gene_end
    

def plot_transcripts(mapper_db, id, id_type = 'Gene Name', fig_width = 15, functional_threshold = 0, transcript_subset = None, sort_by_function = True, coding_color = 'red', noncoding_color = 'white', add_ptms = False, ax = None):
    """
    Given a gene ID, plot all transcripts associated with a given gene. Coding regions are highlighted, by default, in red, and noncoding regions are white. Exons always appear in rank order/direction of translation, even if transcripts are on the reverse strand.
    
    Parameters
    ----------
    id: string
        Name/ID of the gene/transcript of interest. If transcript ID, will plot all transcripts associated with the same gene as the given transcript.
    id_type: string, optional
        Indicates the type of ID given. Options are 'Gene ID', 'Gene Name', or 'Transcript ID'. The default is 'Gene Name'.
    fig_width: float, optional
        Indicates the width of the figure to plot. The default is 15.
    functional_threshold: float, optional
        Indicates the minimum TRIFID functional score required for transcript to be plotted. 0 will plot all transcripts, 1 is the maximum score. The default is 0.
    sort_by_function: bool, optional
        If True, will sort transcripts by TRIFID functional score. If False, transcripts are not sorted. The default is True.
    coding_color: string, optional
        Color to use to indicate the coding region of the exons. Default is 'red'.
    noncoding_color: string, optional
        Color to use to indicate the noncoding region of the exons. Default is 'white'.
    add_ptms: bool, optional
        If True, will add a line where PTMs are located. Default is False.

    Returns
    -------
    ax: matplotlib axis
        figure containing all transcripts plotted

    """
            
    from matplotlib import patches
    import matplotlib.pyplot as plt

    if id_type == 'Gene Name':
        gene_id = mapper_db.get_gene_id(id)
    elif id_type == 'Gene ID':
        gene_id = id
    else:
        raise ValueError("id_type must be 'Gene Name' or 'Gene ID'")


    transcript_ids = mapper_db.get_transcript_ids(gene_id, id_type = 'Gene ID', sort_by_function=True, TRIFID_threshold=functional_threshold)
    if len(transcript_ids) == 0:
        raise ValueError(f'No transcripts found for {id} of type {id_type}')

    #establish plot range 
    ax = None 
    num_transcripts = len(transcript_ids)
    if ax is None:
        fig, ax = plt.subplots(figsize = (6,num_transcripts))
    ax.set_ylim([0,num_transcripts])
    #transcript box
    query = """SELECT Gene_start, Gene_end, Strand FROM genes WHERE Gene_stable_ID = ?"""
    gene_start, gene_end, strand = mapper_db.conn.execute(query, (gene_id,)).fetchone()
    ax.set_xlim(gene_start-2500, gene_end+2500)
    noncoding_color = 'lightgray'
    coding_color = 'red'
    #for each transcript associated with gene, plot exons along gene axis
    row = 1
    for tid in transcript_ids:

        
        #get location of cds start and cds end
        query = """SELECT exons.Gene_Start, exons.Gene_End FROM exons 
            JOIN transcript_exon ON exons.Exon_stable_ID = transcript_exon.Exon_stable_ID
            WHERE transcript_exon.Transcript_stable_ID = ?"""
        tid_exons = mapper_db.conn.execute(query, (tid,)).fetchall()

        gene_cds_start, gene_cds_end = getGenomicCodingRegion(mapper_db, tid, strand)

        #add transcript name and type (canonical/alternative) to plot
        if strand == 1:
            ax.annotate(tid, (gene_start-1500, num_transcripts - row +0.6), ha = 'right', va = 'center')
            #ax.annotate(trans_type, (gene_start-1500, num_transcripts - row +0.4), ha = 'right', va = 'center')
        else:
            ax.annotate(tid, (gene_end+1500, num_transcripts - row +0.6), ha = 'right', va = 'center')
            #ax.annotate(trans_type, (gene_end+1500, num_transcripts - row +0.4), ha = 'right', va = 'center')

        #add line to indicate the gene
        ax.plot([gene_start, gene_end], [num_transcripts-row+0.5, num_transcripts-row+0.5], c = 'k')

        #move through each exon associated with transcript and plot each exon as a rectangle.
        for exons in tid_exons:
            #extract starting and end positions of exon
            fiveprime = exons[0]
            threeprime = exons[1]

            #check if exon is fully noncoding, fully coding, or if cds start/stop exists in exon. Plot exon accordingly
            if threeprime < gene_cds_start or fiveprime > gene_cds_end: #fully noncoding
                rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = noncoding_color, edgecolor = 'black', zorder = 2)
                ax.add_patch(rect)
            elif fiveprime >= gene_cds_start and threeprime <= gene_cds_end: #fully coding
                rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = coding_color, edgecolor = 'black', zorder = 2)
                ax.add_patch(rect)
            else:#partially coding
                noncoding_rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = noncoding_color, edgecolor = 'black', zorder = 2)
                ax.add_patch(noncoding_rect)
                
                if fiveprime < gene_cds_start:
                    fiveprime = gene_cds_start
                if threeprime > gene_cds_end:
                    threeprime = gene_cds_end
                
                noncoding_rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = coding_color, edgecolor = 'black', zorder = 3)
                ax.add_patch(noncoding_rect)
            #rect = patches.Rectangle((fiveprime,num_transcripts - row +0.2), threeprime - fiveprime, 0.6, facecolor = 'red', edgecolor = 'black', zorder = 2)
            #ax.add_patch(rect)

        row = row + 1

    ax.axis('off')
    if strand == -1:
        ax.invert_xaxis()





    if add_ptms:
        #isolate relevant ptms
        ptms_in_region = self.ptm_coordinates[self.ptm_coordinates['Chromosome/scaffold name'] == gene['Chromosome/scaffold name']]
        ptms_in_region = ptms_in_region[ptms_in_region['Strand'] == gene['Strand']]
        ptms_in_region = ptms_in_region[(ptms_in_region['Gene Location (hg38)'] >= gene['Gene start (bp)']) & (ptms_in_region['Gene Location (hg38)'] <= gene['Gene end (bp)'])]
        #add ptms to plot
        if isinstance(add_ptms, list):
            for ptm in add_ptms:
                loc = ptms_in_region.loc[ptms_in_region['Source of PTM'] == ptm, 'Gene Location (hg38)'].values[0]
                mod_type = ptms_in_region.loc[ptms_in_region['Source of PTM'] == ptm, 'Modification Class'].values
                if 'Phosphorylation' in mod_type:
                    color = 'gold'
                elif 'Glycosylation' in mod_type:
                    color = 'lightpink'
                elif 'Methylation' in mod_type or 'methyl' in mod_type:
                    color = 'lightblue'
                elif 'Ubiquitination' in mod_type:
                    color = 'orange'
                elif 'Acetylation' in mod_type or 'acetyl' in mod_type:
                    color = 'lightgreen'
                elif 'Sumoylation' in mod_type:
                    color = 'brown'
                else:
                    color = 'lightgrey'
                ax.axvline(loc, c = color, lw = 0.5, zorder = 10)
        else:
            for i,row in ptms_in_region.iterrows():
                loc = row['Gene Location (hg38)']
                mod_type = row['Modification Class']
                if 'Phospho' in mod_type:
                    color = 'gold'
                elif 'Glyco' in mod_type:
                    color = 'lightpink'
                elif 'Methyl' in mod_type or 'methyl' in mod_type:
                    color = 'lightblue'
                elif 'Ubiquitination' in mod_type:
                    color = 'orange'
                elif 'Acetyl' in mod_type or 'acetyl' in mod_type:
                    color = 'lightgreen'
                elif 'Sumo' in mod_type:
                    color = 'brown'
                else:
                    color = 'lightgrey'
                ax.axvline(loc, c = color, lw = 0.5, zorder = 10)

    return ax


def plot_protein_info(exons, ptms, domains = None):
    fig = go.Figure()

    # Add exon rectangles
    for exon in exons:
        #color based on constitutive or not
        if exon[-1] is not None:
            if exon[2] == 1:
                fillcolor = 'gray'
            else:
                fillcolor = 'darksalmon'

            add_exon(fig, exon[-2], exon[-1], fillcolor = fillcolor, 
                     trace_label = f"Exon: {exon[0]}<br>Rank: {exon[3]}<br>Range: {round(exon[-2],2)}-{round(exon[-1],2)}<br>Constitutive: {exon[2] == 1}", 
                     name = f"Exon{exon[3]}")


    #add text labels tp left of plot to indicate domain and exon rows
    fig.add_annotation(
            x=-1,
            y=0.45,
            text="Exons",
            showarrow=False,
            font=dict(size=16), xanchor = 'right'
    )


    # Add PTM points
    for i, ptm in ptms.iterrows():
        #color based on whether constitutive
        if ptm['ptm_conservation_score'] == 1:
            color = 'gray'
        else:
            color = 'darksalmon'

        #check if canonical position is provided if so, include in trace
        if 'Canonical_Position' in ptm:
            trace = f"PTM Site: {ptm['Residue']}{ptm['Position']}<br>Modification: {ptm['Modification_Class']}<br>Fraction of Isoforms: {round(ptm['ptm_conservation_score'], 2)}<br>Canonical Position: {ptm['Canonical_Position']}<br>Conserved Flank: {ptm['Conserved_Flank']}"
        else:
            trace = f"PTM Site: {ptm['Residue']}{ptm['Position']}<br>Modification: {ptm['Modification_Class']}<br>Fraction of Isoforms: {round(ptm['ptm_conservation_score'], 2)}"
        fig = add_ptm(fig, ptm['Position'], trace_label=trace, color=color)

    fig.add_annotation(
        x=-1,
        y=0.51,
        text="PTMs",
        showarrow=False,
        font=dict(size=16),
        xanchor = 'right'
    )

    #if domains are provided
    if domains:
        #add domains underneath exons
        for d in domains:
            if len(d) == 4:
                fig = add_domain(fig, d[2], d[3], d[1])

        fig.add_annotation(
            x=-1,
            y=0.3,
            text="Domains",
            showarrow=False,
            font=dict(size=16),
            align="left",
            xanchor = 'right'
        )

    fig.update_layout(
        xaxis_title="Protein Position",
        yaxis_visible=False,
        shapes=[],
        height=300,
        width=700,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def add_exon(fig, start, stop, fillcolor = 'red', trace_label = '', name = ''):
    fig.add_shape(
        type="rect",
        x0=start, x1=stop,
        y0=0.4, y1=0.5,
        line=dict(color="black"),
        fillcolor=fillcolor,
    )
    
    # Add a scatter trace covering the rectangle for hover and legend

    center = stop - (stop - start) / 2
    fig.add_trace(go.Scatter(
        x=[center],
        y=[0.4],
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color="black"),
        mode="markers",
        name=name,
        hovertemplate=trace_label,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[start, stop, stop, start, start],
        y=[0.4, 0.4, 0.5, 0.5, 0.4],
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color="black"),
        mode="lines",
        name=name,
        showlegend=False
    ))
    return fig

def add_ptm(fig, position, trace_label='', color='grey'):
    fig.add_trace(go.Scatter(
        x=[position],
        y=[0.51],
        mode="markers",
        marker=dict(size=9, color=color),
        hovertemplate=trace_label,
        name = '',
        showlegend=False,
    ))
    return fig

def add_domain(fig, start, stop, domain_name):
    fig.add_shape(
        type="rect",
        x0=start, x1=stop,
        y0=0.25, y1=0.35,
        line=dict(color="blue"),
        fillcolor="lightblue",
    )

    fig.add_trace(go.Scatter(
        x=[start, stop, stop, start, start],
        y=[0.25, 0.25, 0.35, 0.35, 0.25],
        fill="toself",
        fillcolor="lightblue",
        line=dict(color="blue"),
        mode="lines+text",
        name=domain_name,
        showlegend=False
    ))
    return fig

def getGenomicCodingRegion(mapper_db, tid, strand):
    query = """SELECT Relative_CDS_Start, Relative_CDS_Stop FROM transcripts WHERE Transcript_stable_ID = ?"""
    cds_start, cds_stop = mapper_db.conn.execute(query, (tid,)).fetchone()

    query = """SELECT Exon_stable_ID, Transcript_start, Transcript_end from transcript_exon
        WHERE Transcript_stable_ID = ?
        AND Transcript_start <= ?
        AND Transcript_end >= ?"""
    coding_start_exon = mapper_db.conn.execute(query, (tid, cds_start, cds_start)).fetchone()
    coding_stop_exon = mapper_db.conn.execute(query, (tid, cds_stop, cds_stop)).fetchone()

    if strand == 1:
        query = """SELECT Gene_Start From exons WHERE Exon_stable_ID = ?"""
        gene_cds_start = mapper_db.conn.execute(query, (coding_start_exon[0],)).fetchone()[0] + (cds_start - coding_start_exon[1])
        gene_cds_end = mapper_db.conn.execute(query, (coding_stop_exon[0],)).fetchone()[0] + (cds_stop - coding_stop_exon[1])
    else:
        query = """SELECT Gene_End From exons WHERE Exon_stable_ID = ?"""
        gene_cds_end = mapper_db.conn.execute(query, (coding_start_exon[0],)).fetchone()[0] - (cds_start - coding_start_exon[1])
        gene_cds_start = mapper_db.conn.execute(query, (coding_stop_exon[0],)).fetchone()[0] - (cds_stop - coding_stop_exon[1])

    return gene_cds_start, gene_cds_end