import analysis.irb_13384.coh_config as config
import os
import tools.general_tools as gt
import SimpleITK as sitk
import tools.image_processing as ip
import tools.registration as reg
from tools import data_io as dio
import shutil

data_io= dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)
import numpy as np



def dilate_segmentation(img, n_pixels):
    """
    Dilates segmentation by n_pixels in 2D and by one voxel in z direction
    """
    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelRadius(n_pixels)
    img_dilated_np = sitk.GetArrayFromImage(img)
    # -- 1) Record which slices contain segmentations
    slice_segm_list = []
    for slice_ix in range(img.GetDepth()):
        slice_img_dilated_np = sitk.GetArrayFromImage(img[:, :, slice_ix])
        # -- check if any segmentation label in this slice
        if np.sum(slice_img_dilated_np) > 0:
            slice_segm_list.append(slice_ix)

    for slice_ix in range(img.GetDepth()):
        # -- if current slice does not have segmentation label but next slice has
        if (not slice_ix in slice_segm_list) and (slice_ix + 1 in slice_segm_list):
            slice_img_dilated = filter.Execute(img[:, :, slice_ix + 1])
        # -- if current slice does not have segmentation label but previous slice has
        elif (not slice_ix in slice_segm_list) and (slice_ix - 1 in slice_segm_list):
            slice_img_dilated = filter.Execute(img[:, :, slice_ix - 1])
        else:
            slice_img_dilated = filter.Execute(img[:, :, slice_ix])
        slice_img_dilated_np = sitk.GetArrayFromImage(slice_img_dilated)
        img_dilated_np[slice_ix, :, :] = sitk.GetArrayFromImage(slice_img_dilated)
    img_dilated = sitk.GetImageFromArray(img_dilated_np)
    img_dilated.SetDirection(img.GetDirection())
    img_dilated.SetOrigin(img.GetOrigin())
    img_dilated.SetSpacing(img.GetSpacing())
    return img_dilated


#missing_dces = ['30', '31', '40', '48', '50', '55']
missing_dces = ['109'] # -> require segmentation check

for subject_id in missing_dces: #data_io.bids_layout.unique('subject'):
    sessions = data_io.bids_layout.get(target='session', subject=subject_id, return_type='id')
    for session in sessions:
        print("=== Processing subject '%s', session '%s'"%(subject_id, session))
        #== Get DCE img
        dce_files = data_io.get_image_files(subject=subject_id, session=session, modality='DCE',
                                            processing='original')
        print(dce_files)
        if len(dce_files)==1:
            path_to_dce_file = dce_files[0]
            #== extract last image from DCE sequence
            path_to_dce_last_sequence = data_io.create_dce_analysis_path(
                                                                    subject=subject_id, session=session,
                                                                    modality='DCE', other='last-sequence',
                                                                    create=True, extension='nii')
            dce_img     = sitk.ReadImage(path_to_dce_file)
            dce_tp_last = reg.extract_tp(dce_img, -1)
            sitk.WriteImage(dce_tp_last, path_to_dce_last_sequence)
            #print(path_to_dce_last_sequence)

            for processing in ['bratumia']:
                #== register T1wPost-3D to DCE last (rigid)
                t1wpost_3d_files = data_io.get_image_files(subject=subject_id, session=session, modality='T1wPost-3D',
                                                        registration='rigid', other='withskull', name='standard',
                                                           processing=processing)
                t1wpost_files = data_io.get_image_files(subject=subject_id, session=session, modality='T1wPost',
                                                        registration='rigid', other='withskull', name='standard',
                                                        processing=processing
                                                        )
                if len(t1wpost_3d_files)==1 or len(t1wpost_files)==1:
                    if len(t1wpost_3d_files)==1:
                        path_to_t1wpost_file = t1wpost_3d_files[0]
                        modality = 'T1wPost-3D'
                    elif len(t1wpost_files)==1:
                        path_to_t1wpost_file = t1wpost_files[0]
                        modality = 'T1wPost'
                    break

            # == register T1wPost-3D to DCE last (rigid)
            output_prefix = os.path.join(os.path.dirname(data_io.create_registered_image_path(
                                                                                subject=subject_id, session=session,
                                                                                modality=modality, other='last-sequence',
                                                                                create=True, extension='nii')),
                                         'T1wPost-reg-to-DCE_')

            reg.register_ants_synquick(fixed_img=path_to_dce_last_sequence,
                                       moving_img=path_to_t1wpost_file,
                                       output_prefix=output_prefix,
                                       registration='r', fixed_mask=None)

            #== apply transformation to tumor segmentation

            segmentation_file_bratumia_all = data_io.create_segmentation_image_path(subject=subject_id, session=session, modality='tumorseg',
                                                         segmentation='all', other='bratumia', create=False,
                                                         processing='bratumia', extension='mha', registration='rigid')


            if os.path.exists(segmentation_file_bratumia_all) :
                segmentation_file = segmentation_file_bratumia_all

                print("Found segmentation file: '%s'" % segmentation_file)

                path_to_seg_for_dce = data_io.create_dce_analysis_path(subject=subject_id, session=session,
                                                                           modality='DCE',
                                                                           segmentation='tumor',
                                                                           other='for-dce', extension='mha')
                path_to_trafo_translation = output_prefix + '0DerivedInitialMovingTranslation.mat'
                path_to_trafo_rigid = output_prefix + '1Rigid.mat'

                reg.ants_apply_transforms(input_img=segmentation_file,
                                          reference_img=path_to_dce_last_sequence,
                                          output_file=path_to_seg_for_dce,
                                          transforms=[path_to_trafo_rigid, path_to_trafo_translation])

                tumor_seg = sitk.Cast(sitk.ReadImage(path_to_seg_for_dce), sitk.sitkUInt8)

                # extract segmentation for Enhancing Tumor
                tumor_seg_enhancing = ip.merge_segmentation_labels(tumor_seg, label_map={1 : 7,
                                                                                         0 : [1, 2, 3, 4, 5, 6]})
                tumor_seg_mask_dilated = dilate_segmentation(tumor_seg_enhancing, 5)

                path_to_seg_for_dce_enhancing = data_io.create_dce_analysis_path(subject=subject_id, session=session,
                                                                                   modality='DCE',
                                                                                   segmentation='tumor',
                                                                                   other='enhancing-for-dce',
                                                                                   extension='mhd')

                path_to_seg_for_dce_dilated = data_io.create_dce_analysis_path(subject=subject_id, session=session,
                                                                                   modality='DCE',
                                                                                   segmentation='tumor',
                                                                                   other='enhancing-for-dce-dilated',
                                                                                   extension='mhd')
                sitk.WriteImage(sitk.Cast(tumor_seg_mask_dilated, sitk.sitkUInt16), path_to_seg_for_dce_enhancing)
                sitk.WriteImage(sitk.Cast(tumor_seg_mask_dilated, sitk.sitkUInt16), path_to_seg_for_dce_dilated)

                #=== copy dce file in output folder (in DCEanalysis)
                dce_file_copy_for_dce_analysis = data_io.create_dce_analysis_path( subject=subject_id, session=session,
                                                                                    modality='DCE',
                                                                                    create=True, extension='nii')
                shutil.copy(path_to_dce_file, dce_file_copy_for_dce_analysis)
            else:
                print("-- No segmentation file found")


# Path to DCE file: /Volumes/BRAIN/CART/STRUCT/sub-109/DCEanalysis/ses-2017-04-12/DCE/sub-109_ses-2017-04-12_DCE.nii
# Path to raw mask:
# -          registered: /Volumes/BRAIN/CART/STRUCT/sub-109/DCEanalysis/ses-2017-04-12/DCE/sub-109_ses-2017-04-12_DCE_seg-tumor_enhancing-for-dce.raw
# -          registered & dilated: /Volumes/BRAIN/CART/STRUCT/sub-109/DCEanalysis/ses-2017-04-12/DCE/sub-109_ses-2017-04-12_DCE_seg-tumor_enhancing-for-dce-dilated.raw
# -
# Output files maybe in: /Volumes/BRAIN/CART/STRUCT/sub-109/DCEanalysis/ses-2017-04-12/parameters ?

# dcm2niix results in two different orientations when applied to normal images (3D) and DCE (4D) images:
# E.g.
# 1) DCE
# TransformMatrix = 1 0 0 0 1 0 0 0 1
# Offset = -100.561 156.904 -2.77315
# CenterOfRotation = 0 0 0
# AnatomicalOrientation = RAI
# 2) T1
# TransformMatrix = -0.999743 0.00872654 -0.0209416 -0.00872462 -0.999962 -0.000182755 -0.0209424 1.1719e-11 0.999781
# Offset = 93.5685 130.842 -77.8291
# CenterOfRotation = 0 0 0
# AnatomicalOrientation = LPI
#
# Prativa's tool applied to DCE gives yet another orientation '
# TransformMatrix = 1 0 0 0 1 0 0 0 1
# Offset = 90.3451 156.781 4.42462
# CenterOfRotation = 0 0 0
# AnatomicalOrientation = RAI

# The 'last sequence' extracted from DCE is however aligned with standard T1 images ... not clear why...!
# Check extraction functions maybe there is a flip somewhere for the ivy gap study.
