from config import *
import socket
import tools.general_tools as gt

hostname = socket.gethostname()

#== path to COH Data data
coh_base_dir_out = "/Volumes/BRAIN/CART"
coh_dir_bids = os.path.join(coh_base_dir_out,'STRUCT')
coh_dir_bratumia = os.path.join(coh_base_dir_out,'BRATUMIA')
coh_dir_analysis = os.path.join(coh_base_dir_out,'ANALYSIS')
coh_dir_analysis_segmentation = os.path.join(coh_dir_analysis,'SEGMENTATION')

coh_dates_upn_map = os.path.join(coh_base_dir_out, "IRB13384.xlsx")

#coh_data_dir = os.path.join(coh_base_dir,'ORIG')
coh_base_dir_in = "/Volumes/WD-EXT_1TB_MacOS_ENC/COH"
coh_base_dir_in = "/Volumes/Macintosh HD-1/Users/mathoncuser/Desktop/DATA/CAR-T-CELL"
coh_base_dir_in = "/Volumes/mathoncuser/Desktop/DATA/CAR-T-CELL"
coh_dir_raw_data = coh_base_dir_in

coh_analysis_dir     = os.path.join(project_path, 'analysis', 'irb_13384')

path_to_coh_bids_config = os.path.join(coh_analysis_dir, 'coh_bids_config.json')

#path_to_id_map = os.path.join(project_path, 'do_not_include_in_git', 'car-t-cell_patient-list_plain.xlsx')

#== output to BRAIN folder
coh_dir_output_repo  = os.path.join(coh_base_dir_out, 'output')
gt.ensure_dir_exists(coh_dir_output_repo)
coh_dir_output_datainfo  = os.path.join(coh_dir_output_repo, 'datainfo')
gt.ensure_dir_exists(coh_dir_output_datainfo)
coh_path_to_metadata_raw_xls = os.path.join(coh_dir_output_datainfo, 'dcm_metadata.xls')
coh_path_to_metadata_raw_pkl = os.path.join(coh_dir_output_datainfo, 'dcm_metadata.pkl')
coh_path_to_metadata_sequences_xls = os.path.join(coh_dir_output_datainfo, 'dcm_metadata_with_sequences.xls')
coh_path_to_metadata_sequences_pkl = os.path.join(coh_dir_output_datainfo, 'dcm_metadata_with_sequences.pkl')
coh_path_to_metadata_sequences_selection_xls = os.path.join(coh_dir_output_datainfo, 'dcm_metadata_with_sequences_selection.xls')
coh_path_to_metadata_sequences_timepoint_summary_xls       = os.path.join(coh_dir_output_datainfo, 'dcm_metadata_with_sequences_timepoint_summary.xls')

coh_dir_output_processed  = os.path.join(coh_dir_output_repo, 'processed')
coh_dir_output_for_nb  = os.path.join(coh_dir_output_repo, 'for_notebook')
coh_dir_output_labelstats_xls = os.path.join(coh_dir_output_repo, 'segmentation_label_stats.xls')
coh_dir_output_labelstats_pkl = os.path.join(coh_dir_output_repo, 'segmentation_label_stats.pkl')

coh_dir_scanned_for_seg_xls = os.path.join(coh_dir_bratumia, 'files_scanned_for_segmentation.xls')
coh_dir_scanned_for_seg_pkl = os.path.join(coh_dir_bratumia, 'files_scanned_for_segmentation.pkl')

# Manually included sequences (no identified atumoatically)
coh_path_manually_included_seqs = os.path.join(coh_base_dir_out, 'patient-info', 'manual_inclusions.xls')
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


