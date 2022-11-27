import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import json
import gzip
import re
import sys
import math
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
def processExons(exon_sequences, exon_info):
	#load exon specific information and compile into one data structure
	exons = pd.merge(exon_info, exon_sequences, on ='Exon stable ID', how = 'left')
	
	print('converting exon information into transcript information')
	#use exon information above to compile transcript sequences
	start = time.time()
	exons = exons.sort_values(by = 'Exon rank in transcript')
	#transcripts = exons.groupby('Transcript stable ID')['seq'].apply(''.join)
	end = time.time()
	print('Elapsed time:', end -start, '\n')
	
	#add gene id to transcripts
	print('Getting exon lengths and cuts')
	#get exon length and cuts
	start = time.time()
	exons["Exon Length"] = exons.apply(exonlength, axis = 1)
	exons = exons.sort_values(by = ['Transcript stable ID', 'Exon rank in transcript'])
	exons['Exon End (Transcript)'] = exons.groupby('Transcript stable ID').cumsum(numeric_only = True)['Exon Length']
	exons['Exon Start (Transcript)'] = exons['Exon End (Transcript)'] - exons['Exon Length']
	#transcripts = pd.DataFrame(transcripts)
	#transcripts['Exon cuts'] = exons.groupby('Transcript stable ID')['Exon End (Transcript)'].apply(list)
	end = time.time()
	print('Elapsed time:', end - start, '\n')
	
	return exons
	
def processTranscripts(transcripts, coding_seqs, exons, APPRIS = None):

	print('Getting transcript sequences from exon data')
	start = time.time()
	sorted_exons = exons.sort_values(by = ['Transcript stable ID','Exon rank in transcript'])
	transcript_sequences = sorted_exons.groupby('Transcript stable ID')['Exon Sequence'].apply(''.join)
	transcript_sequences.name = 'Transcript Sequence'
	sorted_exons['Exon End (Transcript)'] = sorted_exons['Exon End (Transcript)'].astype(str)
	transcript_cuts = sorted_exons.groupby('Transcript stable ID')['Exon End (Transcript)'].apply(','.join)
	transcript_cuts.name = 'Exon cuts'
	transcripts = transcripts.join([transcript_sequences, transcript_cuts])
	end = time.time()
	print('Elapsed time:',end - start,'\n')


	print('Finding coding sequence location in transcript')
	#find coding sequence location in transcript
	start = time.time()
	#add coding sequences to transcript info
	transcripts = pd.merge(transcripts, coding_seqs, right_on = 'Transcript stable ID',left_index = True, how = 'left')
	#cds_start,cds_stop = transcripts.apply(findCodingRegion, axis = 1)
	cds_start, cds_stop = findCodingRegion(transcripts)
	transcripts['Relative CDS Start (bp)'] = cds_start
	transcripts['Relative CDS Stop (bp)'] = cds_stop
	end = time.time()
	print('Elapsed time:',end - start,'\n')
	
	#add appris functional scores if information is provided
	if APPRIS is not None:
		print('Adding APPRIS functional scores')
		start = time.time()
		APPRIS = APPRIS[['transcript_id','ccdsid','norm_trifid_score']]
		APPRIS = APPRIS.rename({'transcript_id':'Transcript stable ID', 'ccdsid':'CCDS ID', 'norm_trifid_score':'TRIFID Score'}, axis = 1)
		transcripts = transcripts.merge(APPRIS, on = 'Transcript stable ID', how = 'left')
		end = time.time()
		print('Elapsed time:', end-start,'\n')
	
	print('Getting amino acid sequences')
	#translate coding region to get amino acid sequence
	start = time.time()
	transcripts = transcripts.apply(translate, axis = 1)
	end = time.time()
	
	#return the fraction of transcripts that were successfully translated
	fraction_translated = transcripts.dropna(subset = 'Amino Acid Sequence').shape[0]/transcripts.shape[0]
	print(f'Fraction of transcripts that were successfully translated: {round(fraction_translated, 2)}')
	print('Elapsed time:',end -start,'\n')
	
	#indicate whether transcript is canonical
	trim_translator = config.translator['Uniprot Canonical']
	transcripts = transcripts.merge(config.translator, on = 'Transcript stable ID', how = 'left')
	
	transcripts.index = transcripts['Transcript stable ID']
	transcripts = transcripts.drop('Transcript stable ID', axis = 1)
	
	return transcripts



def getProteinInfo(transcripts, genes):
	"""
	Process translator so to get protein specific information (collapse protein isoforms into one row.
	"""
	proteins = config.translator[config.translator['Uniprot Canonical'] == 'Canonical']
	proteins = proteins[['UniProtKB/Swiss-Prot ID','Transcript stable ID']].drop_duplicates()
	proteins = proteins.groupby('UniProtKB/Swiss-Prot ID').agg(','.join)
	proteins.columns = ['Canonical Transcripts']

	#get variants
	variants = config.translator[config.translator['Uniprot Canonical'] == 'Alternative']
	variants = variants[['UniProtKB/Swiss-Prot ID', 'Transcript stable ID']].drop_duplicates()
	variants_grouped = variants.groupby('UniProtKB/Swiss-Prot ID')
	num_variants = variants_grouped.count() + 1
	num_variants.columns = ['Number of Uniprot Isoforms']
	variant_trans = variants_grouped.agg(','.join)
	variant_trans.columns = ['Alternative Transcripts (Uniprot Isoforms)']

	#get the number of alternative isoforms

	#add available transcripts with matching uniprot sequence
	canonical_matches = []
	for trans in proteins['Canonical Transcripts']:
		match = []
		for available in config.available_transcripts:
			if available in trans:
				match.append(available)
		canonical_matches.append(','.join(match))
	proteins['Matched Canonical Transcripts'] = canonical_matches

	#add number of uniprot isoforms to dataframe, replace nan with 1 (only have the canonical)
	proteins = proteins.merge(num_variants, left_index = True, right_index = True, how = 'left')
	proteins.loc[proteins['Number of Uniprot Isoforms'].isna(), 'Number of Uniprot Isoforms'] = 1
	#add alternative transcripts to dataframe
	proteins = proteins.merge(variant_trans, left_index = True, right_index = True, how = 'left')

	#add available transcripts with matching uniprot sequence
	alternative_matches = []
	for trans in proteins['Alternative Transcripts (Uniprot Isoforms)']:
		match = []
		if trans is np.nan:
			alternative_matches.append(np.nan)
		else:
			for available in config.available_transcripts:
				if available in trans:
					match.append(available)
			alternative_matches.append(','.join(match))
	proteins['Matched Alternative Transcripts'] = alternative_matches
	
	
	#add all available alternative transcripts
	alternative_transcripts = config.translator[config.translator['Uniprot Canonical'] != 'Canonical'].groupby('Gene stable ID')['Transcript stable ID'].apply(','.join)

	proteins['Alternative Transcripts (All)'] = alternative_transcripts
    

	return proteins
	
	
def getMultiProteinGenes(genes):
	num_uniprot = config.translator.groupby('Gene stable ID')['UniProtKB/Swiss-Prot ID'].nunique()
	num_uniprot.name = 'Number of Associated Uniprot Proteins'
	genes = genes.join(num_uniprot)
	return genes

def findCodingRegion(transcripts):
	five_cut = []
	three_cut = []
	for i in transcripts.index:
		coding_sequence = transcripts.at[i,'Coding Sequence']
		full_transcript = transcripts.at[i,'Transcript Sequence']
		if full_transcript is not np.nan:
			match = re.search(coding_sequence, full_transcript)
			if match:
				five_cut.append(match.span()[0])
				three_cut.append(match.span()[1])
			else:
				five_cut.append('error:no match found')
				three_cut.append('error:no match found')
		else:
			five_cut.append('error:no available transcript sequence')
			three_cut.append('error:no available transcript sequence')
	
	return five_cut, three_cut

def getMatchedTranscripts(transcripts, update = False):	
	if config.available_transcripts is None or update:
		print('Finding available transcripts')
		start = time.time()
		#get transcripts whose amino acid sequences are identifical in proteomeScoute and GenCode
		seq_align = config.translator.dropna(subset = 'UniProtKB/Swiss-Prot ID')
		seq_align['PS Seq'] = seq_align.apply(get_ps_seq, axis= 1)
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

def getExonSeq(exon, transcripts):
	"""
	Given the processed exon and transcript dataframes 
	"""
	#make sure exon has associated transcript, if not return Missing Transcript Info
	try:
		transcript = transcripts.loc[exon['Transcript stable ID']]
	except KeyError:
		return 'Missing Transcript Info', 'Missing Transcript Info', 'Missing Transcript Info', 'Missing Transcript Info'
	full_aa_seq = transcript['Amino Acid Sequence']
	#If coding sequence or transcript sequence not available for transcript, indicate
	if isinstance(full_aa_seq, float) or transcript['Relative CDS Start (bp)'] == 'error:no match found':
		return 'No coding seq', 'No coding seq', 'No coding seq', 'No coding seq'
	elif transcript['Relative CDS Start (bp)'] == 'error:no available transcript sequence':
		return 'No transcript seq', 'No transcript seq', 'No transcript seq', 'No transcript seq'
	#check to where exon starts in coding sequence (outside of coding sequence, at protein start, with ragged end, or in middle)
	if int(exon['Exon End (Transcript)']) <= int(transcript['Relative CDS Start (bp)']):
		return "5' NCR", "5' NCR", "5' NCR", "5' NCR"
	elif int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']) == 1 or int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']) == 2:
		#for rare case where exon only partially encodes for the starting amino acid
		return full_aa_seq[0]+'*', 'Partial start codon', 'Partial start codon', 'Partial start codon'
	elif exon['Exon Start (Transcript)'] <= int(transcript['Relative CDS Start (bp)']):
		exon_prot_start = 0.0
	else:
		exon_prot_start = (exon['Exon Start (Transcript)'] - int(transcript['Relative CDS Start (bp)']))/3
		
	exon_prot_end = (int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']))/3
	if exon['Exon Start (Transcript)'] > int(transcript['Relative CDS Start (bp)'])+len(transcript['Coding Sequence']):
		return "3' NCR", "3' NCR", "3' NCR", "3' NCR"
	# in some cases a stop codon is present in the middle of the coding sequence: this is designed to catch those cases (also might be good to identify these cases)
	elif exon['Exon Start (Transcript)'] > int(transcript['Relative CDS Start (bp)'])+len(transcript['Amino Acid Sequence'])*3:
		return "3' NCR", "3' NCR", "3' NCR", "3' NCR"
	elif exon_prot_end > float(len(transcript['Amino Acid Sequence'])):
		exon_prot_end= float(len(transcript['Amino Acid Sequence']))
	else:
		exon_prot_end= (int(exon['Exon End (Transcript)']) - int(transcript['Relative CDS Start (bp)']))/3 

	
	if exon_prot_start.is_integer() and exon_prot_end.is_integer():
		aa_seq_ragged = full_aa_seq[int(exon_prot_start):int(exon_prot_end)]
		aa_seq_nr = full_aa_seq[int(exon_prot_start):int(exon_prot_end)]
	elif exon_prot_end.is_integer():
		ragged_start = math.floor(exon_prot_start)
		full_start = math.ceil(exon_prot_start)
		aa_seq_ragged = full_aa_seq[ragged_start]+'-'+full_aa_seq[full_start:int(exon_prot_end)]
		aa_seq_nr = full_aa_seq[full_start:int(exon_prot_end)]
	elif exon_prot_start.is_integer():
		ragged_stop = math.ceil(exon_prot_end)
		full_stop = math.floor(exon_prot_end)
		aa_seq_ragged = full_aa_seq[int(exon_prot_start):full_stop]+'-'+full_aa_seq[ragged_stop-1]
		aa_seq_nr = full_aa_seq[int(exon_prot_start):full_stop]
	else:
		ragged_start = math.floor(exon_prot_start)
		full_start = math.ceil(exon_prot_start)
		ragged_stop = math.ceil(exon_prot_end)
		full_stop = math.floor(exon_prot_end)
		aa_seq_ragged = full_aa_seq[ragged_start]+'-'+full_aa_seq[full_start:full_stop]+'-'+full_aa_seq[ragged_stop-1]
		aa_seq_nr = full_aa_seq[full_start:full_stop]

	return aa_seq_ragged, aa_seq_nr, exon_prot_start, exon_prot_end

def getAllExonSequences(exons, transcripts):
	exon_seqs_ragged = []
	exon_seqs_nr = []
	exon_prot_starts = []
	exon_prot_ends = []
	for e in range(exons.shape[0]):
		exon = exons.iloc[e]
		results = getExonSeq(exon, transcripts)
		exon_seqs_ragged.append(results[0])
		exon_seqs_nr.append(results[1])
		exon_prot_starts.append(results[2])
		exon_prot_ends.append(results[3])
		
	exons['Exon Start (Protein)'] = exon_prot_starts
	exons['Exon End (Protein)'] = exon_prot_ends
	exons['Exon AA Seq (Ragged)'] = aa_seq_ragged
	exons['Exon AA Seq (Full Codon)'] = aa_seq_nr
		
	return exons

def exonlength(row, seq_col = 'Exon Sequence'):
	exon = row[seq_col]
	if exon is np.nan:
		length = row['Exon end in Gene (bp)']-row['Exon start in Gene (bp)']+1
	else:
		length = len(exon)
		
	return length


def translate(row, coding_seq_col = 'Coding Sequence'):
	seq = row[coding_seq_col]
	if seq is np.nan or seq is None:
		row['Translation Errors'] = 'No Coding Sequence'
		row['Amino Acid Sequence'] = np.nan
		return row
	elif len(seq) % 3 != 0 and seq[0:3] != 'ATG':
		row['Translation Errors'] = 'Start codon error (not ATG);Partial codon error'
		row['Amino Acid Sequence'] = np.nan
	elif len(seq) % 3 != 0:
		row['Translation Errors'] = 'Partial codon error'
		row['Amino Acid Sequence'] =np.nan
		#trim sequence explicitly
	elif seq[0:3] != 'ATG':
		row['Translation Errors'] = 'Start codon error (not ATG)'
		row['Amino Acid Sequence'] = np.nan
	else:
		row['Translation Errors'] = np.nan
		#translate
		coding_strand = Seq(seq)
		aa_seq = str(coding_strand.translate(to_stop = True))
		row['Amino Acid Sequence'] = aa_seq
		
	
	return row
	


def get_ps_seq(row):
	uniprot_id = row['UniProtKB/Swiss-Prot ID']
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
		try:
			seq.append(transcripts.loc[gen_id, 'Amino Acid Sequence'])
		except KeyError:		 
			seq.append(np.nan)
		
	return seq
	   
def perfect_align(row):
	
	u = row['PS Seq']
	v = row['GENCODE Seq']
	ans = u==v
	
	return ans