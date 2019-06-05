import os
import pandas as pd

import analysis.irb_13384.coh_config as config
import analysis.irb_13384.coh_helpers as ch

#== (1) parse ALL DICOM data
# This is the most time consuming step ... can take ~1h when reading from network directory
# Other steps will take seconds
df_dcm_raw = ch.parse_dcm_tree(config.coh_dir_raw_data)
#== (1b) alternativel, read metadata
#df_dcm_raw = pd.read_pickle(config.coh_path_to_metadata_raw_pkl)
#== (1c) add data
#new_dirs = ['/Volumes/mathoncuser/Desktop/DATA/CAR-T-CELL/XXXXXXXX']
#df_dcm_raw = ch.add_patients(new_dirs)


# remove patient_id ''125\x00 fil''
df_dcm_raw = df_dcm_raw[df_dcm_raw.patient_id!='125\x00 fil']

#== map MRM ids to UPN ids
df_dcm_upn = ch.map_mrm_to_upn(df_dcm_raw)


df_dcm_upn = df_dcm_upn.drop_duplicates('series_instance_uid', keep='last')

#== identify sequences from dicom metadata
df_seq = ch.identify_sequences(df_dcm_upn)

#== anonymize metadata in table
df_seq_anonym = ch.anonymize_df(df_seq)

#== save anonymized version of summary tables
# df_seq_anonym_summary = df_seq_anonym.set_index(['patient_id', 'study_instance_uid'])[
#         ['study_date', 'time_from_first_visit', 'seqs_per_timepoint', 'anatomic_axial']]
df_seq_anonym_summary = df_seq_anonym.set_index(['patient_id', 'study_instance_uid'])[
         ['study_date', 'time_from_first_visit']]
df_seq_anonym.to_excel(os.path.join(config.coh_base_dir_out, 'sequence_summary.xls'))
df_seq_anonym.set_index(['patient_id', 'study_instance_uid']).sort_index().to_excel(os.path.join(config.coh_base_dir_out, 'sequence_summary_index.xls'))



