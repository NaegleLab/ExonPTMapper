import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import re
import sys
from ExonPTMapper import config
import time
#import swifter

"""
Need 3 file types downloaded from_ensembl (can only download limited information form ensembl)
1) exon_rank: contains both Exon IDs and exon rank
2) exon_sequences: contains both Exon ID and exon sequences
3) coding_sequences: contains the full coding sequence for all transcripts


****Would likely be useful to have code for pulling from ensembl rather than just file types
"""
def processExons(exon_sequences, exon_rank, coding_seqs, unspliced_gene = None):
	#load exon specific information and compile into one data structure
	exons = pd.merge(exon_rank, exon_sequences, left_on = 'Exon stable ID', right_on = 'id', how = 'left')
	
	print('converting exon information into transcript information')
	#use exon information above to compile transcript sequences
	start = time.time()
	exons = exons.sort_values(by = 'Exon rank in transcript')
	transcripts = exons.groupby('Transcript stable ID')['seq'].apply(''.join)
	end = time.time()
	print('Elapsed time:', end -start, '\n')
	
	#add gene id to transcripts
	print('Getting exon lengths and cuts')
	#get exon length and cuts
	start = time.time()
	exons["Exon Length"] = exons.apply(exonlength, axis = 1)
	exons = exons.sort_values(by = ['Transcript stable ID', 'Exon rank in transcript'])
	exons['Exon End (Transcript)'] = exons.groupby('Transcript stable ID').cumsum()['Exon Length']
	exons['Exon Start (Transcript)'] = exons['Exon End (Transcript)'] - exons['Exon Length']
	transcripts = pd.DataFrame(transcripts)
	transcripts['Exon cuts'] = exons.groupby('Transcript stable ID')['Exon End (Transcript)'].apply(list)
	end = time.time()
	print('Elapsed time:', end - start, '\n')
	
	
	print('Finding coding sequence location in transcript')
	#find coding sequence location in transcript
	start = time.time()
	#add coding sequences to transcript info
	transcripts = pd.merge(transcripts, coding_seqs, on = 'Transcript stable ID', how = 'inner')
	#cds_start,cds_stop = transcripts.apply(findCodingRegion, axis = 1)
	cds_start, cds_stop = findCodingRegion(transcripts)
	transcripts['CDS Start'] = cds_start
	transcripts['CDS Stop'] = cds_stop
	
	end = time.time()
	print('Elapsed time:',end - start,'\n')
	
	print('Getting amino acid sequences')
	#translate coding region to get amino acid sequence
	start = time.time()
	transcripts['Amino Acid Seq'] = transcripts.apply(translate, axis = 1)
	end = time.time()
	print('Elapsed time:',end -start,'\n')
	
    
	#add gene id to transcripts dataframe
	transcripts.index = transcripts['Transcript stable ID']
	gene_ids = exons[['Gene stable ID', 'Transcript stable ID']].drop_duplicates()
	gene_ids.index = gene_ids['Transcript stable ID']
	#identify transcript IDs found in both the transcript dataframe and exon dataframe
	overlapping_trans = set(gene_ids.index).intersection(transcripts.index)
	gene_ids = gene_ids.loc[list(overlapping_trans)]
	transcripts.loc[gene_ids.index.values, 'Gene stable ID'] = gene_ids['Gene stable ID']
	
	return exons, transcripts
    
def getGeneInfo(exons, transcripts, gene_seqs):
    """
    After running processExons, get gene-specific information, such as gene sequence, whether it contains multiple exons, whether it codes for protein, etc.
    """
    print('Finding single exon genes')
    start = time.time()
    #get all exon cuts for each transcript
    tmp = transcripts['Exon cuts'].apply(lambda x: x[1:-1].split(','))
    #identify transcripts with single exon cut (only one exon in gene)
    single_exon_transcripts = tmp[tmp.apply(len) == 1].index.values
    transcripts['Single Exon Transcript'] = False
    transcripts.loc[single_exon_transcripts, 'Single Exon Transcript'] = True
    single_exon_genes = transcripts.groupby('Gene stable ID')['Single Exon Transcript'].all()
    gene_seqs['Single Exon Gene'] = single_exon_genes
    end = time.time()
    print('Elapsed Time:',end-start,'\n')
    
    print('Finding genes with at least one transcript with mapped coding sequence')
    start = time.time()
    #identify all transcripts with coding sequence
    coding_transcripts = transcripts[transcripts['coding seq'] != 'Sequenceunavailable']
    #identify genes associated with at least one coding transcript
    coding_genes = coding_transcripts['Gene stable ID'].unique()
    #record genes with coding sequence in genes dataframe
    gene_seqs['Coding Gene'] = False
    coding_genes = [gene for gene in coding_genes if gene in gene_seqs.index.values]
    gene_seqs.loc[coding_genes, 'Coding Gene'] = True
    stop = time.time()
    print('Elapsed time:',end-start,'\n')
    return gene_seqs
    

def getMatchedTransripts(transcripts, update = False):	
	if config.available_transcripts is None or update:
		print('Finding available transcripts')
		start = time.time()
		#get transcripts whose amino acid sequences are identifical in proteomeScoute and GenCode
		seq_align = config.translator.drop(['canonicals'], axis=1)
		seq_align.rename(columns={'Transcript and isoform':'Gencode ID'}, inplace=True)
		seq_align['PS Seq'] = seq_align.apply(get_ps_seq, axis= 1)
		seq_align['PS Query ID'] = seq_align.apply(get_uni_id, axis= 1)
		#seq_align['GENCODE Seq'] = seq_align.apply(get_gencode_seq, transcripts, axis= 1)
		seq_align['GENCODE Seq'] = get_gencode_seq(seq_align, transcripts)
		seq_align['Exact Match'] = seq_align.apply(perfect_align, axis=1)
		perfect_matches = seq_align[seq_align['Exact Match']==True]
		# if the transcript is a perfect match, 
		# then take PTMs assigned to it and track to exon
		
		config.available_transcripts = perfect_matches['Transcript stable ID'].tolist()
		with open(config.processed_data_dir+"available_transcripts.json", 'w') as f:
			json.dump(config.available_transcripts, f, indent=2) 
		end = time.time()
		print('Elapsed time:',end-start, '\n')
	else:
		print('Already have the available transcripts. If you would like to update analysis, set update=True')

def exonlength(row):
	exon = row['seq']
	length = len(exon)
	
	return length

	
def findCodingRegion(transcripts):
	five_cut = []
	three_cut = []
	for i in transcripts.index:
		coding_sequence = transcripts.at[i,'coding seq']
		full_transcript = transcripts.at[i,'seq']
		match = re.search(coding_sequence, full_transcript)
		if match:
			five_cut.append(match.span()[0])
			three_cut.append(match.span()[1])
		else:
			five_cut.append('error:no match found')
			three_cut.append('error:no match found')
	
	return five_cut, three_cut
	
def translate(row):
	seq = row['coding seq']
	
	if seq == 'error1':
		aa_seq = 'error1'
	elif seq == 'error: no start codon matched':
		aa_seq = 'error: no start codon matched'
	elif seq == 'error: not a string':
		aa_seq = 'error: not a string'
	elif seq == 'Sequenceunavailable':
		aa_seq = 'coding sequence unavailable, try old method on seq column'
	else:
		coding_strand = Seq(seq)
		aa_seq = coding_strand.translate(to_stop = True)
	
	return aa_seq
	
def findExonInGene(exons, unspliced_gene):
	start = []
	stop = []
	for i in exons.index:
		gene = exons.loc[i,'Gene stable ID']
		if gene in unspliced_gene.index:
			gene_seq = unspliced_gene.loc[gene].values[0]
			exon_seq = exons.loc[i,'seq']
			match = re.search(exon_seq, gene_seq)
			if match is None:
				start.append('no match')
				stop.append('no match')
			else:
				start.append(match.span()[0])
				stop.append(match.span()[1])
		else:
			start.append('gene not found')
			stop.append('gene not found')
	return start, stop
		
def get_ps_seq(row):
	uniprot_id = row['Uniprot ID']
	seq = config.ps_api.get_sequence(uniprot_id)  # uses base Uniprot ID to get sequence without the isoform info
	return seq
	
def get_uni_id(row):
	if row['Isoform'] == '1':	
		uni_id = row['Uniprot ID']		 
	else:
		info = row['Isoform']
		import re
		preans = info.split('[')[1]
		ans = preans.split(']')[0]
		uni_id = re.sub('-', '.', ans)	# this line will be deleted once P.S. is updated, rn the isoform records are listed as PXXX.2 instead of PXXX-2 
		
	return uni_id
	
def get_gencode_seq(seq_align, transcripts):
	seq = []
	for i in seq_align.index:
		gen_id = seq_align.at[i,'Transcript stable ID']
		if len(transcripts[transcripts['Transcript stable ID']==gen_id]['Amino Acid Seq'].tolist()) == 0:
			seq.append('N/A')
		else:		 
			seq.append(str(transcripts[transcripts['Transcript stable ID']==gen_id]['Amino Acid Seq'].tolist()[0]))
		
	return seq
	   
def perfect_align(row):
	
	u = row['PS Seq']
	v = row['GENCODE Seq']
	ans = u==v
	
	return ans