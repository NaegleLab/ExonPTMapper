import pandas as pd

def checkFrame(exon, transcript, loc, loc_type = 'Gene', strand = 1):
    """
    given location in gene, transcript, or exon, return the location in the frame

    1 = first base pair of codon
    2 = second base pair of codon
    3 = third base pair of codon

    Primary motivation of the function is to determine whether the same gene location in different transcripts is found in the same reading frame

    Parameters
    ----------
    exon: pandas series
        series object containing exon information for exon of interest
    transcript: pandas series
        series object containing transcript information related to exon of interest
    loc: int
        location of nucleotide to check where in frame it is located
    loc_type: string

    """
    if loc_type == 'Gene':
        #calculate location of nucleotide in exon (with 0 being the first base pair of the exon). Consider whether on reverse or forward strand
        if strand == 1:
            loc_in_exon = loc - int(exon['Exon Start (Gene)'])
        else:
            loc_in_exon = exon['Exon End (Gene)'] - loc
        #calculate location of nucleotide in transcript (with 0 being the first base pair of the entire transcript, including UTRs)
        loc_in_transcript = loc_in_exon + int(exon['Exon Start (Transcript)'])
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        frame = (loc_in_transcript - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    elif loc_type == 'Exon':
        #calculate location of nucleotide in transcript (with 0 being the first base pair of the entire transcript, including UTRs)
        loc_in_transcript = loc + int(exon['Exon Start (Transcript)'])
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        frame = (loc_in_transcript - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    elif loc_type == 'Transcript':
        #calculate the location in the reading frame (mod returns 0 if multiple of 3 but want this to indicate first bp of a codon, so add 1)
        frame = (loc - int(transcript['Relative CDS Start (bp)'])) % 3 + 1
    else:
        print("Invalid loc_type. Can only be based on location in 'Gene','Exon', or 'Transcript'")
    return frame