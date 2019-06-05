import analysis.irb_13384.coh_helpers as ch
import analysis.irb_13384.coh_config as config
import os
import pandas as pd


# read manually classified cases
base_dir = config.coh_dir_raw_data
df_manual = pd.read_excel(config.coh_path_manually_included_seqs)
df_manual.study_date = pd.to_datetime(df_manual.study_date)
df_manual['path_to_dir'] = df_manual.rel_path_to_dir.apply(lambda x : os.path.join(base_dir, x))
# merge with automatically classified cases
df_seqs = pd.read_pickle(config.coh_path_to_metadata_sequences_pkl).reset_index()
df_all = df_seqs.append(df_manual, ignore_index=True)
# copy / extract
ch.organize_files(df_seqs=df_all, export_dcm=True, overwrite=False, anonymise=True)
