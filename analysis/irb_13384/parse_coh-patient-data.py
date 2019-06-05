from heapq import merge

from numpy.distutils.system_info import dfftw_threads_info

from tools import dcm_tools
from tools import general_tools as gt

import os
import pandas as pd
import numpy as np

# Mapping dictionary for parsing Series Description Strings in Ivy GAP
series_descr_map = {
    'orientation' : {  'AX' : ['AX', 'AXIAL'],
                       'SAG': ['SAG'],
                       'COR': ['COR']},
    'weighting'   : {  'T1w' : ['T1'],
                       'T2w' : ['T2'],
                       'T2*' : ['T2*']},
    'dimension'   :  ['3D'],
    'sequence'    :  ['GRE', 'FSPGR', 'FSE', 'FRFSE', 'EPI', 'ATNL', 'FLAIR', 'FLASH', 'MAP FA'],
    'contrast'    :  {  'Post' : ['Post', 'GD', 'GAD'],
                        'Pre'  : ['Pre']},
    'keyword'     : {  'DSC' : ['perfusion', 'perf'],
                       'DCE' : ['dce', 'dynamic'],
                       'ADC' : ['adc', 'apparent'],
                       'DWI' : ['dw'],
                       'DTI' : ['dti']},
    'FA'          : { 'num'  : 'FA[0-9]{1,2}'}
}


path_to_dicom = "/Volumes/WD-EXT_1TB_MacOS_ENC/COH-PatientData_Final"

output_dir    = os.path.join(path_to_dicom, 'dataset_info')
gt.ensure_dir_exists(output_dir)

df = dcm_tools.create_pd_from_dcm_dir(dcm_dir=path_to_dicom, out_dir=output_dir)

# convert study date to date-time format
df.study_date = pd.to_datetime(df.study_date)

df = df.set_index(['patient_id', 'study_instance_uid', 'series_instance_uid'])


# identify sequence information
#df = pd.read_pickle(os.path.join(output_dir, 'data_with_sequences.pkl'))


df_seqs = dcm_tools.identify_sequences(df, series_descr_map, output_dir)


# Create Summary df
summary_df = df_seqs.reset_index().groupby(['patient_id', 'study_instance_uid']).head(1).drop('series_instance_uid', axis=1)
summary_df = summary_df.set_index(['patient_id', 'study_instance_uid'])[
                ['study_date', 'seqs_per_timepoint', 'anatomic_axial']]
summary_df.to_excel(os.path.join(output_dir, 'summary.xls'))


base_input_dir  = path_to_dicom
base_output_dir = "/Volumes/WDEXT/COH_CART/BDIS"


sel = df_seqs

# for index, row in sel.reset_index().iterrows():
#     # path_to_dicom_dir
#     source_dir_subject= row['patient_id']
#     source_dir_study  = row['study_instance_uid']
#     source_dir_series = row['series_instance_uid']
#     path_to_dicom_dir = row['path_to_dir']
#     # BDIS folder structure
#     subject_dir_name = row['patient_id']
#     session_dir_name = 'session_'+row['study_date'].strftime('%Y-%m-%d')
#     acqu_dir_name    = row['sequence_name']
#     file_name        = subject_dir_name + '_' + session_dir_name + '_' + acqu_dir_name
#     path_to_output_folder = os.path.join(base_output_dir,subject_dir_name,session_dir_name, acqu_dir_name)
#     if row.sequence_name in ['T1w', 'T1wPost', 'T2w', 'T2wFLAIR'] and row.orientation=='AX':
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, anonymise=True)
#     elif row.sequence_name in ['DWI', 'ADC', 'DSC', 'DCE']:
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, anonymise=True)
#     elif row.sequence_name in ['T1wMapFA']:
#         file_name = file_name + "_" +  row.FA
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, anonymise=True)
#
