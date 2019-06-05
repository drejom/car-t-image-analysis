import analysis.irb_13384.coh_config as config
from tools import data_io as dio
import analysis.irb_13384.coh_helpers as ch
import tools.general_tools as gt
import os



gt.ensure_dir_exists(config.coh_dir_analysis_segmentation)
data_io= dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)

# This function looks for existing segmentation files and analyzes them
# It gives preferences to files ending in '_p.mha'
df = ch.analyze_segmentations(data_io, subjects=None)

#-- compute total volume
all_segmentation_labels = [col for col in df.columns if col.startswith('bratumia')]
df["bratumia_total_segmented_volume"] = df[all_segmentation_labels].sum(axis=1)
all_tumor_labels = ['bratumia_EnhancingTumor', 'bratumia_Necrosis', 'bratumia_NonEnhancingTumor']
df["bratumia_TotalTumor"] = df[all_tumor_labels].sum(axis=1)
other_tumor_labels = ['bratumia_Necrosis', 'bratumia_NonEnhancingTumor']
df["bratumia_OtherTumor"] = df[other_tumor_labels].sum(axis=1)
#-- save
df.to_excel(os.path.join(config.coh_dir_analysis_segmentation, 'segmentation_stats_single_index.xls'))
df = df.set_index(['subject_id', 'session']).sort_index()
df.to_excel(config.coh_dir_output_labelstats_xls)
df.to_excel(os.path.join(config.coh_dir_analysis_segmentation, 'segmentation_stats.xls'))


# plot segmentation volumes
plot_selection = ['Edema', 'EnhancingTumor', 'NonEnhancingTumor', "Necrosis"]
ch.plot_segmentation_volumes(df, subject_ids=None, # plots all
                             plot_selection = plot_selection,
                             out_dir=os.path.join(config.coh_dir_analysis_segmentation, 'PLOTS'),
                             show=False)