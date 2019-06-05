from heapq import merge

from numpy.distutils.system_info import dfftw_threads_info

from tools import dcm_tools
from tools import general_tools as gt

import os
import pandas as pd
import numpy as np
import shutil

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

base_dir = "/Volumes/WD-EXT_1TB_MacOS_ENC/COH_CART"
path_to_dicom = os.path.join(base_dir, "DOI")
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

#
# base_input_dir  = path_to_dicom
# path_to_dicom = os.path.join(base_dir, "BIDS")
#
# sel = df_seqs
#
#
#
# from do_not_include_in_git.patient_id_to_upn_map import patient_id_to_upn_map
#
# for index, row in sel.reset_index().iterrows():
#     # path_to_dicom_dir
#     source_dir_subject= row['patient_id']
#     source_dir_study  = row['study_instance_uid']
#     source_dir_series = row['series_instance_uid']
#     path_to_dicom_dir = row['path_to_dir']
#     # BDIS folder structure
#     subject_dir_name = str(patient_id_to_upn_map[int(row['patient_id'])])
#     session_dir_name = 'session_'+row['study_date'].strftime('%Y-%m-%d')
#     acqu_dir_name    = row['sequence_name']
#     file_name        = subject_dir_name + '_' + session_dir_name + '_' + acqu_dir_name
#     path_to_output_folder = os.path.join(base_output_dir,subject_dir_name,session_dir_name, acqu_dir_name)
#     if row.sequence_name in ['T1w', 'T1wPost', 'T2w', 'T2wFLAIR'] and row.orientation=='AX':
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, export_dcm=True, anonymise=True)
#     elif row.sequence_name in ['DWI', 'ADC', 'DSC', 'DCE']:
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, export_dcm=True, anonymise=True)
#     elif row.sequence_name in ['T1wMapFA']:
#         file_name = file_name + "_" +  row.FA
#         print("# %s -- %s -- %s"%(subject_dir_name, session_dir_name, file_name))
#         dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name, export_dcm=True, anonymise=True)
#     # Copy DCE params file
#     path_to_dce_param_dir = os.path.join(os.path.dirname(path_to_dicom_dir), 'NII', 'Output_Perfusion_LTKM_NoneDesculp', 'Data')
#     outpath_dce_param_dir = os.path.join(os.path.join(base_output_dir,subject_dir_name,session_dir_name,'DCE-params'))
#     for filename in ['lambda', 'Ktr', 'vp']:
#         path_to_dce_param_file = os.path.join(path_to_dce_param_dir, filename)
#         output_file_name       =  subject_dir_name + '_' + session_dir_name + '_' + filename
#         outpath_dce_param_file = os.path.join(outpath_dce_param_dir, output_file_name)
#         if os.path.exists(path_to_dce_param_file):
#             if not os.path.exists(outpath_dce_param_file):
#                 gt.ensure_dir_exists(os.path.dirname(outpath_dce_param_file))
#                 shutil.copyfile(path_to_dce_param_file, outpath_dce_param_file)
#
#

#
# a=dcm_tools.create_pd_from_dcm_dir("/Volumes/WD-EXT_1TB_MacOS_ENC/COH_CART/UPN117/2016.03.09/S5",output_dir)
#
# import pydicom
# path = '/Volumes/WD-EXT_1TB_MacOS_ENC/COH_CART/UPN117/2016.03.09/S5/1.3.12.2.1107.5.2.36.40273.2015121610122794576685646.dcm'
# ds = pydicom.dcmread(path, force=True)