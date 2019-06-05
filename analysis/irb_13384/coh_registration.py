import tools.registration as reg
import os


base_data_dir = "/Volumes/WD-EXT_1TB_MacOS_ENC/COH_CART/BDIS"
target_data_dir = "/Volumes/WD-EXT_1TB/COH_CART/BDIS"
subject_id      = "10171606"

for subject_id in os.listdir(base_data_dir):
    if not subject_id.startswith('.'):
        reg.register_patient(base_data_dir, subject_id, ref_seq=['T1w-3D', 'T1w', 'T1wPost-3D', 'T1wPost'] , session_prefix='session_')
        reg.copy_registered_files(os.path.join(base_data_dir, subject_id),
                                  os.path.join(base_data_dir, '..', 'BDIS_REG', subject_id))



#
# path_to_T1_file    = "/Volumes/WDEXT/COH_CART/BDIS/10171606/session_2017-09-22/T1w/10171606_session_2017-09-22_T1w.nii"
# path_to_DCE_file   = "/Volumes/WDEXT/COH_CART/BDIS/10171606/session_2017-09-22/DCE/10171606_session_2017-09-22_DCE.nii"
#
#
# ref_img = sitk.ReadImage(path_to_T1_file)
# dce_img = sitk.ReadImage(path_to_DCE_file)
# dce_tp_0 = extract_tp(dce_img, 0)
# transform = register_simple(ref_img, dce_tp_0)
#
# dce_tp_file_name_reg_to_T1w = "dce_to_T1.nii"
# save_transform_and_image(transform, ref_img, dce_tp_0, os.path.dirname(path_to_T1_file),
#                          dce_tp_file_name_reg_to_T1w)
#
#
#
# split_and_register_dce(path_to_T1_file, path_to_DCE_file, os.path.dirname(path_to_DCE_file))


#
#
# path_to_T1_file    = "/Volumes/WDEXT/COH_CART/BDIS/10171606/session_2017-09-22/T1w/10171606_session_2017-09-22_T1w.nii"
# path_to_DCE_file   = "/Volumes/WDEXT/COH_CART/BDIS/10171606/session_2017-09-22/DCE/10171606_session_2017-09-22_DCE.nii"
# path_to_lambda_file = '/Volumes/WDEXT/COH_CART/BDIS/10171606/session_2017-09-22/DCE/lambda.raw'
#
# ref_img = sitk.ReadImage(path_to_T1_file)
# dce_img = sitk.ReadImage(path_to_DCE_file)
#
# register_dce_param_map(dce_img, ref_img, path_to_raw_file=path_to_lambda_file,
#                        output_path=os.path.dirname(path_to_lambda_file), name='dce_param_lambda')
#
#






