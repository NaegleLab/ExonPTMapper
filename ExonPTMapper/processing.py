import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import re
import sys
from ExonPTMapper import config
import dask.dataframe as dd
import dask
dask.config.set(scheduler='processes')

import time
#import swifter

"""
Need 3 file types downloaded from_ensembl (can only download limited information form ensembl)
1) exon_rank: contains both Exon IDs and exon rank
2) exon_sequences: contains both Exon ID and exon sequences
3) coding_sequences: contains the full coding sequence for all transcripts


****Would likely be useful to have code for pulling from ensembl rather than just file types
"""
def processExons(exon_sequences, exon_rank, coding_seqs, unspliced_gene = None, PROCESSES = 1, parallel_approach = 'dask'):
	#load exon specific information and compile into one data structure
	print('Creating exon dataframe')
	start = time.time()
	exons = pd.merge(exon_rank, exon_sequences, left_on = 'Exon stable ID', right_on = 'id', how = 'left')
	end = time.time()
	print('Elapsed time:', end -start, '\n')
	
	print('converting exon information into transcript information')
	#use exon information above to compile transcript sequences
	start = time.time()
	exons = exons.sort_values(by = ['Exon rank in transcript'])
	transcripts = exons.groupby('Transcript stable ID')['seq'].apply(''.join)
	end = time.time()
	print('Elapsed time:', end -start, '\n')
	
	
	print('Getting exon lengths and cuts')
	#get exon length and cuts
	start = time.time()
	exons["Exon Length"] = exons['seq'].apply(len)
	exons = exons.sort_values(by = ['Transcript stable ID', 'Exon rank in transcript'])
	exons['Exon End (Transcript)'] = exons.groupby('Transcript stable ID').cumsum()['Exon Length']
	exons['Exon Start (Transcript)'] = exons['Exon End (Transcript)'] - exons['Exon Length']
	transcripts = pd.DataFrame(transcripts)
	exons['Exon End (Transcript)'] = exons['Exon End (Transcript)'].astype(str)
	transcripts['Exon cuts'] = exons.groupby('Transcript stable ID')['Exon End (Transcript)'].apply(','.join)
	end = time.time()
	print('Elapsed time:', end - start, '\n')

	if PROCESSES > 1:
		#get dask dataframe
		dd_exons = dd.from_pandas(exons, npartitions = PROCESSES)
		dd_trans = dd.from_pandas(transcripts, npartitions = PROCESSES)
		
		print('Finding coding sequence location in transcript')
		#find coding sequence location in transcript
		start = time.time()
		#add coding sequences to transcript info
		dd_trans = dd_trans.merge(coding_seqs, on = 'Transcript stable ID', how = 'inner')
		#cds_start,cds_stop = transcripts.apply(findCodingRegion, axis = 1)
		cds = dd_trans.apply(findCodingRegion, axis = 1, meta = (None, 'object')).apply(pd.Series, index=['CDS Start', 'CDS Stop'], meta={'CDS Start':int,'CDS Stop': int}).compute()
		dd_trans['CDS Start'] = cds['CDS Start']
		dd_trans['CDS Stop'] = cds['CDS Stop']
		
		end = time.time()
		print('Elapsed time:',end - start,'\n')
		
		
		print('Getting amino acid sequences')
		#translate coding region to get amino acid sequence
		start = time.time()
		dd_trans['Amino Acid Seq'] = dd_trans['coding seq'].apply(translate, meta = ('coding seq','str')).compute()
		end = time.time()
		print('Elapsed time:',end -start,'\n')
		
		print('Getting gene info')
		#add gene id to transcripts dataframe
		start = time.time()
		dd_trans.index = dd_trans['Transcript stable ID']
		gene_ids = exons[['Gene stable ID', 'Transcript stable ID']].drop_duplicates()
		gene_ids.index = gene_ids['Transcript stable ID']
		#identify transcript IDs found in both the transcript dataframe and exon dataframe
		overlapping_trans = set(gene_ids.index).intersection(dd_trans.index)
		gene_ids = gene_ids.loc[list(overlapping_trans)]
		dd_trans.loc[gene_ids.index.values, 'Gene stable ID'] = gene_ids['Gene stable ID']
		end = time.time()
		print('Elapsed time:',end -start,'\n')
		
		return dd_exons, dd_transcripts
	else:
		
		
		
		print('Finding coding sequence location in transcript')
		#find coding sequence location in transcript
		start = time.time()
		#add coding sequences to transcript info
		transcripts = pd.merge(transcripts, coding_seqs, on = 'Transcript stable ID', how = 'inner')
		#cds_start,cds_stop = transcripts.apply(findCodingRegion, axis = 1)
		cds = transcripts.apply(findCodingRegion, axis = 1).apply(pd.Series, index=['CDS Start', 'CDS Stop'])
		transcripts['CDS Start'] = cds['CDS Start']
		transcripts['CDS Stop'] = cds['CDS Stop']
		
		end = time.time()
		print('Elapsed time:',end - start,'\n')
		
		print('Getting amino acid sequences')
		#translate coding region to get amino acid sequence
		start = time.time()
		transcripts['Amino Acid Seq'] = transcripts['coding seq'].apply(translate)
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

def findCodingRegion(transcript):
	coding_sequence = transcript['coding seq']
	full_transcript = transcript['seq']
	match = re.search(coding_sequence, full_transcript)
	if match:
		return match.span()[0], match.span()[1]
	else:
		return 'error:no match found', 'error:no match found'

"""
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
"""

def translate(seq):
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