import analysis.irb_13384.coh_config as config
from tools import data_io as dio
import analysis.irb_13384.coh_helpers as ch
import tools.general_tools as gt
import os
import pandas as pd

gt.ensure_dir_exists(config.coh_dir_analysis_segmentation)
#-- read

path_to_seg = '/Volumes/BRAIN/CART/ANALYSIS/SEGMENTATION/segmentation_stats_single_index.xls'
df_seg = pd.read_excel(path_to_seg)
df_seg.session = pd.to_datetime(df_seg.session)


#-- read car-t dates
df_dates = pd.read_excel(config.coh_dates_upn_map, header=0)

# clean colum headers
df_dates.columns = [str(col).strip() for col in df_dates.columns]
# convert dates to datetime
df_dates.MRI1 = pd.to_datetime(df_dates.MRI1)
df_dates.MRI2 = pd.to_datetime(df_dates.MRI2)
df_dates.MRI3 = pd.to_datetime(df_dates.MRI3)
df_dates.MRI4 = pd.to_datetime(df_dates.MRI4)

df_dates_wide = pd.wide_to_long(df_dates, stubnames=['MRI', 'RANO'], i=['UPN'], j='cart_administration').reset_index()
df_dates_wide = df_dates_wide[['UPN', 'DCE', 'ARM', 'MRI', 'RANO', 'cart_administration']]
df_dates_wide.columns = ['subject_id', 'has_dce', 'trial_arm', 'session', 'rano', 'cart_administration']
df_dates_wide.to_excel('/Volumes/BRAIN/CART/IRB13384_wide.xls')


# merge both dataframes

# merge and keep keys from both dfs
df_merged = pd.merge(df_seg, df_dates_wide, on=['subject_id', 'session'], how='outer')
#remove dfs where no date row
df_merged = df_merged[~df_merged.session.isna()]
df_merged.to_excel('/Volumes/BRAIN/CART/ANALYSIS/SEGMENTATION/segmentation_stats_cart_dates.xls')
df_merged.set_index(['subject_id', 'session']).sort_index().to_excel('/Volumes/BRAIN/CART/ANALYSIS/SEGMENTATION/segmentation_stats_cart_dates_uid-date-index.xls')


df_sel = df_merged[~df_merged.cart_administration.isna()]

df_sel.set_index(['subject_id', 'session']).sort_index().to_excel('/Volumes/BRAIN/CART/ANALYSIS/SEGMENTATION/segmentation_stats_cart_dates_uid-date-index_selection.xls')



df_sel_wide_enhanc = df_sel.reset_index().pivot(index='subject_id', columns='cart_administration', values=['bratumia_EnhancingTumor', 'bratumia_Edema', 'trial_arm', 'rano'])
df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', '2-1')] =  \
                                                  df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', 2)] \
                                                - df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', 1)]
df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', '3-2')] =  \
                                                  df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', 3)] \
                                                - df_sel_wide_enhanc.loc[slice(None),('bratumia_EnhancingTumor', 2)]

df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', '2-1')] =  \
                                                  df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', 2)] \
                                                - df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', 1)]
df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', '3-2')] =  \
                                                  df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', 3)] \
                                                - df_sel_wide_enhanc.loc[slice(None),('bratumia_Edema', 2)]

df_sel_wide_enhanc.sort_index().sort_index(axis=1).to_excel('/Volumes/BRAIN/CART/ANALYSIS/SEGMENTATION/segmentation_selection_enhancing.xls')


