import os
import socket

hostname = socket.gethostname()

#== automatically infer project path
project_path = os.path.dirname(__file__)

data_path = os.path.join(project_path, 'data')
output_path = os.path.join(project_path, 'output')

#== selected paths to atlas data
atlas_dir_orig      = os.path.join(project_path, 'data', 'atlas', 'mni_icbm152_nlin_sym_09a')
atlas_dir_proc      = os.path.join(project_path, 'data', 'atlas', 'mni_icbm152_processed')
path_to_atlas_t1    = os.path.join(atlas_dir_orig,'mni_icbm152_t1_tal_nlin_sym_09a.mnc')
path_to_atlas_brainmask   = os.path.join(atlas_dir_orig, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.mnc')
path_to_atlas_ventricles  = os.path.join(atlas_dir_proc, 'mni_icbm152_labels_ventricles.mha')
path_to_atlas_t1_skullstripped         = os.path.join(atlas_dir_proc, 'mni_icbm152_t1_skullstripped.mha')
path_to_atlas_t1_flipped_skullstripped = os.path.join(atlas_dir_proc, 'mni_icbm152_t1_flipped_skullstripped.mha')
path_to_atlas_t1_flipped               = os.path.join(atlas_dir_proc, 'mni_icbm152_t1_flipped.mha')
path_to_atlas_brainmask_flipped        = os.path.join(atlas_dir_proc, 'mni_icbm152_brainmask_flipped.mha')
path_to_atlas_ventricles_flipped       = os.path.join(atlas_dir_proc, 'mni_icbm152_labels_ventricles_flipped.mha')

atlas_dir_sri     = os.path.join(project_path, 'data', 'atlas', 'sri24')
path_to_sri_atlas_t1            = os.path.join(atlas_dir_sri,'SRI24_T1_RAI.mha')
path_to_sri_atlas_t1_orient     = os.path.join(atlas_dir_sri,'SRI24_T1_RAI_oriented.mha')
path_to_sri_atlas_t1_orient_skullstripped     = os.path.join(atlas_dir_sri,'SRI24_T1_RAI_oriented_SkullStripped.mha')
path_to_sri_atlas_labels_tissue        = os.path.join(atlas_dir_sri,'SRI24_tissue_labels_RAI.mha')
path_to_sri_atlas_labels_ventricles        = os.path.join(atlas_dir_sri,'SRI24_tissue_labels_RAI_ventricles.mha')


atlas_dir_whs_rat_atlas                 = os.path.join(project_path, 'data', 'atlas', 'whs_sd_rat')
path_to_whs_rat_atlas_labels            = os.path.join(atlas_dir_whs_rat_atlas, 'WHS_SD_rat_atlas_v2_RIP.nii.gz')
path_to_whs_rat_atlas_t1                = os.path.join(atlas_dir_whs_rat_atlas, 'WHS_SD_rat_T2star_v1.01_RIP.nii.gz')
path_to_whs_rat_atlas_brainmask         = os.path.join(atlas_dir_whs_rat_atlas, 'WHS_SD_v2_brainmask_bin_RIP.nii.gz')
path_to_whs_rat_atlas_t1_skullstripped   = os.path.join(atlas_dir_whs_rat_atlas, 'WHS_SD_rat_T2star_v1_skullstripped.nii')
path_to_whs_rat_atlas_labels_ventricles  = os.path.join(atlas_dir_whs_rat_atlas, 'WHS_SD_rat_labels_ventricles.mha')

#== Value maps

label_tissue_map_bratumia = {
    1 : 'CSF',
    2 : 'GrayMatter',
    3 : 'WhiteMatter',
    4 : 'Necrosis',
    5 : 'Edema',
    6 : 'NonEnhancingTumor',
    7 : 'EnhancingTumor',
    #--- extended, see bratumia merge map
    8 : 'T1c',
    9 : 'T2'
}


label_tissue_map_manual = {
    8 : 'T1c',
    9 : 'T2'
}

bratumia_merge_map_T1c = {
    8 : [4, 6, 7],
    0 : [1, 2, 3, 5]
}

bratumia_merge_map_T2 = {
    9 : [4, 5, 6, 7],
    0 : [1, 2, 3]
}


#== path to IVY GAP data
path_environment = os.environ.copy()

if hostname=="danabl-CB-ISTB":
    ants_path = "/home/danabl/software/ANTs_build/bin"
    ants_script_path = "/home/danabl/software/ANTs/Scripts"
    path_environment["PATH"] = ants_path + ":" + ants_script_path + ":" +\
                               path_environment["PATH"]
    path_environment["ANTSPATH"] = ants_path
elif hostname=='daniel-pc':
    ants_path = "/home/danabl/software/ANTs_build/bin"
    ants_script_path = "/home/danabl/software/ANTs/Scripts"
    path_environment["PATH"] = ants_path + ":" + ants_script_path + ":" + \
                               path_environment["PATH"]
    path_environment["ANTSPATH"] = ants_path
else:
    ants_script_path = "/Users/dabler/software/ANTs/scripts"
