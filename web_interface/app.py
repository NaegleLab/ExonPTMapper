import streamlit as st
import hydralit_components as hc
import pandas as pd
import sqlite3
from sqlite3 import Connection

import sys
sys.path.append('..')
import plotting
st.set_page_config(layout='wide')

@st.cache_resource()
def get_connection(path):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.
    NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
    """
    return sqlite3.connect(path, check_same_thread=False)
#if 'connection' not in st.session_state:
#    st.session_state['db'] = sql_interface.mapper_db('../mapper.db')
#if 'plotter' not in st.session_state:
#    plotter = plotting.plotter('../mapper.db')

conn = get_connection('../SQL_Database/mapper.db')
plotter = plotting.plotter(conn)

menu_data = [
    {'label':"Protein"},
    {'label':"Splice Event"}
]

menu = hc.nav_bar(menu_definition = menu_data, sticky_nav = True, sticky_mode = 'pinned')


if menu == "Protein":
    cols = st.columns(2)
    text_id = cols[0].text_input("Enter Protein ID/Name:")
    id_type = cols[1].selectbox("Select ID Type:", ["UniProt", "Name"])
    #show_constitutive = st.checkbox('Include Constitutive PTMs')
    if text_id:
        #table = st.session_state['db'].get_canonical_ptm_table(text_id, id_type, show_constitutive)
        #st.write(table)


        st.write("Let's visualize the protein (exons, PTMs, and domains)! Please pick an isoform to visualize:")
        isoforms = plotter.get_isoform_ids(text_id, id_type)
        canonical_isoform = plotter.get_canonical_isoform(text_id, id_type = id_type)

        #append indicator which is canonical
        isoforms = [i + " (Canonical)" if i == canonical_isoform else i for i in isoforms]
        isoform_id = st.selectbox("Select Isoform:", isoforms).split(' ')[0]    #make sure to remove canonical tag

        #output transcripts associated with isoform
        transcripts = plotter.get_transcript_ids(isoform_id, id_type = 'Isoform')
        st.write('Transcript IDs: ' + ', '.join(transcripts))

        exons = plotter.get_exon_info(isoform_id, id_type = 'Isoform')
        ptms = plotter.get_ptm_table(isoform_id, include_constitutive = True)
        if isoform_id == canonical_isoform:
            domains = plotter.get_domain_info(isoform_id.split('-')[0], id_type = 'UniProt')
            #domains = None
        else:
            domains = None
            
        fig = plotting.plot_protein_info(exons, ptms)
        for d in domains:
            fig = plotting.add_domain(fig, d[2],d[3], d[1])
        st.plotly_chart(fig)
elif menu == "Splice Event":
    st.title("Splice Event Analysis")
    
    #get splice event file
    event_type = st.selectbox("Select splice event type you would like to analyze:", 
        ['SE', 'A5SS', 'A3SS', 'RI', 'MXE'], index = 0)
    alpha = st.slider("Select significance threshold (FDR):", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    min_dpsi = st.slider("Select minimum absolute inclusion level difference:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    splice_event_file = st.file_uploader("Upload rMATS event file (tsv format):", type=['txt'])

    if splice_event_file is not None:
        splice_events = pd.read_csv(splice_event_file, sep = '\t')
        #filter for significant events
        splice_events = splice_events[(splice_events['FDR'] <= alpha) & (abs(splice_events['IncLevelDifference']) >= min_dpsi)]
        st.success('Splice event file uploaded')

        st.write(f"File contains {splice_events.shape[0]} {event_type} events. Please indicate the gene you would like to analyze:")
        gene = st.text_input("Enter Gene Name:")
        if gene:
            #filter splice events for gene
            data_sub = splice_events[splice_events['geneSymbol'] == gene]
            if data_sub.shape[0] == 0:
                st.warning(f"No significant {event_type} events found for gene {gene}.")
            elif data_sub.shape[0] == 1:
                st.write(f"Found 1 {event_type} event for gene {gene}. Visualizing now.")
                rmats_event = data_sub.squeeze()
                fig = plotter.plot_splice_event(rmats_event, event_type = event_type)
                st.plotly_chart(fig)
            else:
                st.write(f"Found {data_sub.shape[0]} {event_type} events for gene {gene}. Please select the event you would like to visualize:")
                st.write(data_sub[['ID', 'exonStart_0base', 'exonEnd', 'upstreamES', 'upstreamEE', 'downstreamES', 'downstreamEE', 'IncFormLen', 'SkipFormLen', 'IncLevelDifference']])
                event_ids = data_sub['ID'].to_list()
                event_id = st.selectbox("Select Event ID:", event_ids)

                visualize_type = st.selectbox("Type of visualization:", ['Splice Event', 'Transcript View'])

                if event_id:
                    rmats_event = data_sub[data_sub['ID'] == event_id].squeeze()
                    st.write(f"Visualizing {event_type} event {event_id} for gene {gene}")

                    if visualize_type == 'Transcript View':
                        functional_threshold = st.slider("Select minimum TRIFID score required to show transcript:", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
                        spliced_region, upstream_region, downstream_region, chromosome, strand = plotting.extract_event_info(rmats_event, event_type = event_type)
                        ax = plotter.plot_transcripts(gene, id_type = 'Name', functional_threshold = functional_threshold, sort_by_function=True)
                        ax.axvspan(spliced_region[0], spliced_region[1], color='green', alpha=0.3, label='Spliced Region')
                        ax.axvspan(upstream_region[0], upstream_region[1], color='blue', alpha=0.3, label='Upstream Region')
                        ax.axvspan(downstream_region[0], downstream_region[1], color='blue', alpha=0.3, label='Downstream Region')
                    else:
                        fig = plotter.plot_splice_event(rmats_event, event_type = event_type)
                        st.plotly_chart(fig)
