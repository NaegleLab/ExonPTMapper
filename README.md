# Interacting with PTM-splicing data with SQL

This branch of ExonPTMapper catalogues work done to convert the .csv files generated from the main branch (genes, exons, ptm_info, etc.) to an SQL database. The branch is broken into three main folders:
1. ExonPTMapper: contains the python modules still needed from the main branc.
2. SQL_Database: all code to generate and interface with local SQL database storing data from .csv files. The following modules exist:
	- `construct_sql_tables.py`: functions and script for generating SQL tables from .csv files and appending to SQL database called `mapper.db`
	-`sql_interface.py`: code to run SQL queries and extract specific data associated with SQL database
	- `report.ipynb` + `assessment.py`: functions and juypter notebook for performing some quick tests to compare .csv approach to SQL approach
3. web_interface: code to run streamlit app and visualize PTMs within exons, proteins, and splice events.

## Generating SQL tables

To generate SQL tables,
1. Make sure config file in '/ExonPTMapper/' points to folder containing .csv files generated from main branch
2. Run script: `python construct_sql_tables.py`
3. You should now have a database file called `mapper.db` saved in the SQL_Database folder

## Running streamlit app

To run the streamlit app,
1. Install streamlit and hydralit_components (`pip install streamlit`, `pip install hydralit-components`)
2. Navigate to '/web_interface/' folder
3. Run streamlit with `streamlit run app.py`. This should open a local host in your web browser that can interact with the SQL database


