Code still in progress, but general idea for package is as follows:

3 main python files:

1. config: contains basic information, such as where to find saved files and the 
transcript to uniprot translator
2. processing: code to take ensemble information and convert into easily usable csv
file, one for exon-level information, one for transcript-level information, and one 
for gene-level information. Ideally this only needs to be run to update with new
ensemble updates
3. mapping: contains python class called PTM_mapper, which contains the bulk of the
functions used for PTM analysis. As it stands, will contain functions both for processing
PTM information and for general visualization/analysis. In the future, may look to split this
into two classes, one for mapping and one for analysis/plotting