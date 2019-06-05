from tools import data_io as dio
import analysis.irb_13384.coh_config as config
import pandas as pd
import numpy as np
import os

data_io = dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)


df = pd.DataFrame()
count = 0
for subject_id in data_io.bids_layout.unique('subject'):
    # for subject_id in bids_rest:
    for session in data_io.bids_layout.get(target='session', subject=subject_id, processing='original',
                                           return_type='id'):
        modalities = data_io.bids_layout.get(target='modality', subject=subject_id, session=session,
                                             processing='original', return_type='id')
        print(subject_id, session, modalities)
        df.loc[count, 'subject_id'] = subject_id
        df.loc[count, 'session'] = session
        for modality in ['T1w', 'T1w-3D', 'T1wPost', 'T1wPost-3D', 'T2w', 'T2wFLAIR', 'T2wPD', 'DCE']:
            has_modality = modality in modalities
            df.loc[count, modality] = has_modality
        count = count+1

df = df.set_index(['subject_id', 'session']).sort_index()

df.to_excel(os.path.join(config.coh_base_dir_out, 'sequences_in_struct.xls'))

df[np.logical_not(df.DCE)].to_excel(os.path.join(config.coh_base_dir_out, 'sequences_wo_DCE.xls'))