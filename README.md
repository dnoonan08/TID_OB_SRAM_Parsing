# ECON SRAM OB Error Parsing

This repo contains scripts for parsing and categorizing ECON OB SRAM errors.

Functions for parsing the data in the json files are located in [parse_sc_data.py](https://github.com/dnoonan08/TID_OB_SRAM_Parsing/blob/main/parse_sc_data.py "parse_sc_data.py").

### Most useful functions:
 - **merge_jsons**: This is designed for merging link capture data from stream comparison tests back into a single json file.  The link captures were split out into separate json files during the January 2025 TID testing to fix memory leak issues.  This is run by simply giving it the name of the "top-level" json file output by pytest.  It then looks for a corresponding file for each iteration of the stream comparison test, with the same name structure in the same directory.  It will merge data into a single json file, and save it with the same name as the initial json, but with the directory changed from `json_files` to `merged_jsons` (if `json_files` is not in the path, or you wish to change the input and output directory names, there are `old_dir_name` and `new_dir_name` arguments)
 - **checkErr**: takes a json file from the TID testing (after merging) and parses out all of the link capture data. Returns two dataframes, one containing total counts of number of errors of different types at each voltage, second containing the data from every packet across all voltages
 - **parse_sram_errors_per_packet**: The "workhorse"  of all of it, called by checkErr function.  Goes through a json file, finding all stream comparison tests, and parsing the link capture data for packets, and classifying each packet based on what type of error it had