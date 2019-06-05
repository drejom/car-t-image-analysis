import analysis.irb_13384.coh_config as config
import pandas as pd
import os

from tools import data_io as dio
from tools import general_tools as gt
import analysis.irb_13384.coh_helpers as ch

#== Step 1) SCAN DIRECTORY TO FIND CASES THAT CAN BE SEGMENTED

df_segm = ch.generate_bratumia_input(n_per_file=100, selection='new')


#== Step 2) run segmentations in bratumia using above batch-file


#== Step 3) copy segmentations into file structure on network drive
data_io = dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)

import glob
bratumia_source_base_dirs = glob.glob(os.path.join(config.coh_dir_bratumia, 'Batch_2019_5_29_13_53'))

for base_dir in bratumia_source_base_dirs:
    for folder in os.listdir(base_dir):
        if not folder.endswith('csv') and not folder.startswith('.'):
            subject_id, session = ch.parse_session_folder_name_bratumia(folder)
            #-- PATHS TO BRATUMIA RESULTS FILES
            bratumia_folder_path = os.path.join(base_dir, folder)
            try:
                ch.copy_bratumia_to_bids(data_io, bratumia_folder_path, subject_id, session, folder,
                                         modalities=["T1w", "T2w", "T1wPost", "T2wFLAIR"],
                                         copy=True, overwrite=False)
            except:
                print("Something went wrong when copying directory '%s'"%bratumia_folder_path)


# Sequences still missing for
# T2 AX TSE  in /Volumes/Macintosh HD-1/Users/mathoncuser/Desktop/DATA/CAR-T-CELL/165<XXX>.2017.01.18
# T2 FLAIR in /Volumes/Macintosh HD-1/Users/mathoncuser/Desktop/DATA/CAR-T-CELL/165<XXX>.2016.10.24
# and sequences listed in excel file ...
