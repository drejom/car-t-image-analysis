import pandas as pd
import os
import shutil

import analysis.irb_13384.coh_config as config
import tools.dcm_tools as dct
import tools.general_tools as gt
from tools import data_io as dio
import tools.image_processing as ip
import SimpleITK as sitk
import matplotlib.pylab as plt

def parse_dcm_tree(base_dir=config.coh_dir_raw_data):
    # == parse ALL DICOM data
    df = dct.create_pd_from_dcm_dir(dcm_dir=base_dir)
    # -- remove 'null', nan
    df = df[~df.patient_id.isna()]
    df = df[df.patient_id != 'null']
    # -- save
    df.to_excel(config.coh_path_to_metadata_raw_xls)
    df.to_pickle(config.coh_path_to_metadata_raw_pkl)
    print("Results saved to '%s'." % config.coh_path_to_metadata_raw_xls)
    print("!! These files contain patient names, and MRM ids !!")
    return df


def add_patients(dcm_dirs):
    df = pd.DataFrame()
    for dir in dcm_dirs:
        print("== Parsing new directory '%s'"%dir)
        df_new = dct.create_pd_from_dcm_dir(dcm_dir=dir)
        df = df.append(df_new)
    # remove nans
    df = df[~df.patient_id.isna()]
    df = df[df.patient_id != 'null']
    # drop duplicate series
    df = df.drop_duplicates('series_instance_uid', keep='last')
    # remove patient_id ''125\x00 fil''
    df = df[df.patient_id != '125\x00 fil']
    # convert patient ids to int
    df.patient_id = df.patient_id.astype(int)
    # merge with existing data so that original data remains and new data if added
    datainfo = pd.read_pickle(config.coh_path_to_metadata_raw_pkl)
    try:
        datainfo = datainfo.reset_index()
    except:
        pass
    datainfo.append(df)
    datainfo = datainfo.drop_duplicates('series_instance_uid', keep='last')
    # save updated metadata
    datainfo.to_excel(config.coh_path_to_metadata_raw_xls)
    datainfo.to_pickle(config.coh_path_to_metadata_raw_pkl)
    print("Results saved to '%s'." % config.coh_path_to_metadata_raw_xls)
    print("!! These files contain patient names, and MRM ids !!")
    return datainfo



def map_mrm_to_upn(df_mrm):
    df_mrm.patient_id = df_mrm.patient_id.astype(int)
    # create patient id - UPN map
    #coh_cart_info = pd.read_excel(config.path_to_id_map)
    #id_map_pd = coh_cart_info[['Name', 'MRM', 'UPN']].dropna()
    coh_cart_info = pd.read_excel(config.coh_dates_upn_map)
    id_map_pd = coh_cart_info[['MRM', 'UPN']].dropna()
    id_map_pd.UPN = id_map_pd.UPN.astype(int)
    id_map_pd.MRM = id_map_pd.MRM.astype(int)
    id_map_pd = id_map_pd[['MRM', 'UPN']]
    id_map_pd.columns = ['patient_id', 'upn']
    # merge into mrm df
    df_upn = pd.merge(df_mrm, id_map_pd, on='patient_id')
    # check if all mapped
    n_mrm_before = len(df_mrm.patient_id.unique())
    n_mrm_after  = len(df_upn.patient_id.unique())
    if n_mrm_before != n_mrm_after:
        print("MRM ids included: ", df_upn.patient_id.unique())
        print("From %i original MRN ids, only %i could be mapped into UPN ids."%(n_mrm_before, n_mrm_after))
        diff = set(df_mrm.patient_id.values).difference(set(df_upn.patient_id.values))
        print("The following UPN ids could not be mapped: %s" %diff)
        print("Please check the mapping table located at '%s'."%config.coh_dates_upn_map)
    return df_upn

def anonymize_df(df_in):
    df_out = df_in.copy(deep=True).reset_index()
    cols_tb_removed = ['patient_name', 'path_to_dir', 'mrm']
    for col in cols_tb_removed:
        if col in df_out.columns:
            df_out = df_out.drop(col, axis=1)
    return df_out

def identify_sequences(df_dcm_info):
    #== pre processing
    # -- convert study date to date-time format
    df_dcm_info.study_date = pd.to_datetime(df_dcm_info.study_date)
    try:
        df_dcm_info = df_dcm_info.reset_index()
    except:
        pass
    df_dcm_info.set_index(['patient_id', 'study_instance_uid', 'series_instance_uid'])
    # -- compute time difference between imaging series
    df_dcm_info['time_from_first_visit'] = df_dcm_info.study_date - df_dcm_info.groupby(['patient_id']).study_date.min()

    # == identify sequences
    # Mapping dictionary for parsing Series Description Strings in COH data
    series_descr_map = {
        'orientation': {'AX': ['AX', 'AXIAL'],
                        'SAG': ['SAG'],
                        'COR': ['COR']},
        'weighting': {'T1w': ['T1'],
                      'T2w': ['T2'],
                      'T2*': ['T2*']},
        # 'dimension'   :  ['3D'],
        'dimension': {'3D': ['MPR', '3D', 'MPRAGE']},
        'sequence': ['GRE', 'FSPGR', 'FSE', 'FRFSE', 'EPI', 'ATNL', 'FLAIR', 'FLASH', 'MAP FA'],
        'contrast': {'Post': ['Post', 'GD', 'GAD', '+C'],
                     'Pre': ['Pre']},
        'keyword': {'DSC': ['perfusion', 'perf'],
                    'DCE': ['dce', 'dynamic'],
                    'ADC': ['adc', 'apparent'],
                    'DWI': ['dw'],
                    'DTI': ['dti'],
                    'PD': ['PD'],
                    'SUB': ['SUB']},
        'FA': {'num': 'FA[0-9]{1,2}'}
    }
    # Run identification -> stores the results
    df_seqs = dct.identify_sequences(df_dcm_info, series_descr_map)
    df_seqs.to_excel(config.coh_path_to_metadata_sequences_xls)
    df_seqs.to_pickle(config.coh_path_to_metadata_sequences_pkl)

    # write summary into to excel
    # 1) per sequence
    df_seqs_selection = df_seqs[
        ['upn', 'study_date', 'patient_name', 'protocol_name', 'series_description', 'orientation', 'sequence_name',
         'path_to_dir']]
    df_seqs_selection.set_index(['upn', 'study_date']).sort_index().to_excel(config.coh_path_to_metadata_sequences_selection_xls)
    # 2) per time point
    summary_df = df_seqs.reset_index().groupby(['patient_id', 'study_instance_uid']).head(1).drop('series_instance_uid',
                                                                                                  axis=1)
    # summary_df = summary_df.set_index(['patient_id', 'study_instance_uid'])[
    #     ['study_date', 'time_from_first_visit', 'seqs_per_timepoint', 'anatomic_axial']]
    summary_df = summary_df.set_index(['patient_id', 'study_instance_uid'])[
             ['study_date', 'time_from_first_visit']]
    summary_df.to_excel(config.coh_path_to_metadata_sequences_timepoint_summary_xls)
    return df_seqs


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def check_for_output_folder(path_to_folder):
    if os.path.exists(path_to_folder):
        path_to_folder = path_to_folder+"_2"
    else:
        gt.ensure_dir_exists(path_to_folder)



def generate_bratumia_input(n_per_file=10, selection='all'):
    data_io = dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)
    gt.ensure_dir_exists(config.coh_dir_bratumia)
    df = pd.DataFrame()
    count = 0
    for subject_id in data_io.bids_layout.unique('subject'):
        # for subject_id in bids_rest:
        for session in data_io.bids_layout.get(target='session', subject=subject_id, processing='original',
                                               return_type='id'):
            modalities = data_io.bids_layout.get(target='modality', subject=subject_id, session=session,
                                                 processing='original', return_type='id')
            print(subject_id, session, modalities)

            # create table
            df.loc[count, 'subject_id'] = subject_id
            df.loc[count, 'session'] = session
            df.loc[count, 'modalities'] = ', '.join(modalities)

            if ((('T2w' in modalities) or ('T2wPD' in modalities)) and ('T2wFLAIR' in modalities) and (
                    ('T1w' in modalities) or ('T1w-3D' in modalities)) and
                    (('T1wPost' in modalities) or ('T1wPost-3D' in modalities))):
                try:
                    # -- T1
                    if 'T1w-3D' in modalities:
                        T1_mod = 'T1w-3D'
                    elif 'T1w' in modalities:
                        T1_mod = 'T1w'
                    path_to_T1 = data_io.bids_layout.get(subject=subject_id, session=session, modality=T1_mod,
                                                         processing='original', extensions='nii')[0].filename
                    # -- T1c
                    if 'T1wPost-3D' in modalities:
                        T1c_mod = 'T1wPost-3D'
                    elif 'T1wPost' in modalities:
                        T1c_mod = 'T1wPost'
                    path_to_T1c = data_io.bids_layout.get(subject=subject_id, session=session, modality=T1c_mod,
                                                          processing='original', extensions='nii')[0].filename
                    # -- T2
                    if 'T2w' in modalities:
                        T2w_mod = 'T2w'
                    elif 'T2wPD' in modalities:
                        T2w_mod = 'T2wPD'
                    path_to_T2 = data_io.bids_layout.get(subject=subject_id, session=session, modality=T2w_mod,
                                                         processing='original', extensions='nii')[0].filename
                    # -- FLAIR
                    path_to_FLAIR = data_io.bids_layout.get(subject=subject_id, session=session, modality='T2wFLAIR',
                                                            processing='original', extensions='nii')[0].filename
                    # -- OUTFILE
                    out_path = subject_id + '_' + session
                    # -- write to file

                    df.loc[count, 'status'] = 'ready for segmentation'
                    df.loc[count, 'path_to_T1'] = path_to_T1
                    df.loc[count, 'path_to_T1c'] = path_to_T1c
                    df.loc[count, 'path_to_T2'] = path_to_T2
                    df.loc[count, 'path_to_FLAIR'] = path_to_FLAIR
                    df.loc[count, 'out_path'] = out_path
                except:
                    print("Problem identifying files")
                    df.loc[count, 'status'] = 'nii missing'
            else:
                print("Not all modalities available for subject '%s', session '%s'" % (subject_id, session))
                df.loc[count, 'status'] = 'modality missing'
            # check if already segmented
            path_to_tumor_seg = data_io.create_registered_image_path(subject=subject_id, session=session,
                                                                     modality='tumorseg',
                                                                     segmentation='tumor', other='bratumia',
                                                                     processing='bratumia',
                                                                     extension='mha', create=False)
            if os.path.exists(path_to_tumor_seg):
                segmented = True
            else:
                segmented = False
            df.loc[count, 'segmented'] = segmented

            count = count + 1

    df.to_excel(config.coh_dir_scanned_for_seg_xls)
    df.to_pickle(config.coh_dir_scanned_for_seg_pkl)
    df_for_bratumia = df[~df.path_to_T1.isna()].reset_index()
    df_tb_reviewed = df[df.path_to_T1.isna()].reset_index()
    df_for_bratumia.set_index(['subject_id', 'session']).sort_index().to_excel(os.path.join(config.coh_dir_bratumia, 'files_ready_for_segmentation.xls'))
    df_tb_reviewed.set_index(['subject_id', 'session']).sort_index().to_excel(os.path.join(config.coh_dir_bratumia, 'files_to_be_reviewed_for_segmentation.xls'))

    bratumia_columns = ['path_to_T1', 'path_to_T1c', 'path_to_T2', 'path_to_FLAIR', 'out_path']

    if selection == 'all':
        df_sel =  df_for_bratumia
    else:
        df_sel =  df_for_bratumia[df_for_bratumia.segmented == False]

    df_sel[bratumia_columns].to_csv(os.path.join(config.coh_dir_bratumia,'to_segment.csv'), index=False, header=False)

    for sublist in chunks(df_sel.subject_id.unique(), n_per_file):
        file_name = "batch_ids_" + "-".join(sublist) + ".csv"
        selection_pd = df_sel[df_sel.subject_id.isin(sublist)]
        selection_pd[bratumia_columns].to_csv(os.path.join(config.coh_dir_bratumia, file_name), index=False,
                                              header=False)
    print("Saved files to %s"%config.coh_dir_bratumia)
    return df_for_bratumia

def organize_files(df_seqs=None, export_dcm=True, overwrite=False, anonymise=True):
    data_io = dio.DataIO(config.coh_dir_bids, config.path_to_coh_bids_config)
    if df_seqs is None:
        df_seqs = pd.read_pickle(config.coh_path_to_metadata_sequences_pkl).reset_index()

    df_seqs.orientation = df_seqs.orientation.astype(str)
    #df_seqs['subject_id'] = df_seqs.patient_id
    sel = df_seqs
    for index, row in sel.iterrows():
        # path_to_dicom_dir
        # source_dir_subject= str(row['patient_name'])
        # source_dir_study  = row['study_instance_uid']
        # source_dir_series = row['series_instance_uid']
        # path_to_dicom_dir = os.path.join(config.coh_dir_raw_data, source_dir_subject, source_dir_study, source_dir_series)
        path_to_dicom_dir = row.path_to_dir
        if row.study_date.to_datetime64() in df_seqs[
                                             df_seqs.upn == row.upn].study_date.sort_values().unique():
            subject_id = str(row.upn)
            session = dio.create_session_name(row.study_date)
            modality = str(row.sequence_name)
            orientation = row.orientation
            print("Processing subject '%s', %s, %s"%(subject_id, session, modality))
            path_to_file = data_io.create_image_path(subject=subject_id, session=session, modality=modality,
                                                     create=False)
            path_to_output_folder = os.path.dirname(path_to_file)
            file_name = os.path.basename(path_to_file)
            if modality in ['T1w', 'T1wPost', 'T2w', 'T2wFLAIR'] and 'AX' in orientation:
                if row.dimension == '3D':
                    modality = modality + '-3D'
                    path_to_file = data_io.create_image_path(subject=subject_id, session=session, modality=modality)
                    path_to_output_folder = os.path.dirname(path_to_file)
                    file_name = os.path.basename(path_to_file)
                print("# %s, %s, %s " % (path_to_dicom_dir, path_to_file, path_to_output_folder))
                gt.ensure_dir_exists(path_to_output_folder)
                dct.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name,
                                             anonymise=anonymise, export_dcm=export_dcm, overwrite=overwrite)
            elif modality in ['T2wPD']:
                print("# %s, %s, %s " % (path_to_dicom_dir, path_to_file, path_to_output_folder))
                gt.ensure_dir_exists(path_to_output_folder)
                dct.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name,
                                             anonymise=anonymise, export_dcm=export_dcm, overwrite=overwrite,
                                             t2pd=True)
            # elif modality in ['DWI', 'ADC', 'DSC', 'DCE']:
            elif modality in ['DCE']:
                print("# %s " % (file_name))
                check_for_output_folder(path_to_output_folder)
                dct.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name,
                                             anonymise=anonymise, export_dcm=export_dcm, overwrite=overwrite)
            # elif modality in ['T1wMapFA']:
            #     file_name = file_name + "_" + row.FA
            #     print("# %s " % (file_name))
            #     gt.ensure_dir_exists(path_to_output_folder)
            #     dcm_tools.convert_dcm_folder(path_to_dicom_dir, path_to_output_folder, file_name,
            #                                 anonymise=False, export_dcm=EXPORT_DCM, overwrite=OVERWRITE)
            sel.loc[index, 'path_to_nii'] = os.path.join(path_to_output_folder, file_name)
    # sel.to_excel(os.path.join(config.coh_dir_output_datainfo, 'selection_bids.xls'))
    # sel.to_pickle(os.path.join(config.coh_dir_output_datainfo, 'selection_bids.pkl'))


def parse_session_folder_name_bratumia(name):
    patient_id, session = name.split('_')
    return patient_id, session


def copy_bratumia_to_bids(data_io, bratumia_dir, subject_id, session, folder, modalities=["T1w", "T2w", "T1wPost", "T2wFLAIR"],
                          copy=True, overwrite=False):
    modality_map = {"T1w" : "T1", "T1w-3D" : "T1w",
                    "T2w" : "T2",
                    "T1wPost" : "T1C", "T1wPost-3D" : "T1C",
                    "T2wFLAIR" : "Flair"}
    bratumia_registered = os.path.join(bratumia_dir,  'Registration', 'Registered')
    bratumia_registered_masked = os.path.join(bratumia_dir,  'Registration', 'Masked')
    bratumia_registered_segmentation = os.path.join(bratumia_dir,  'RegisteredSegmentations')
    #-- registered, not skullstripped
    for modality in modalities:
        path_to_source = os.path.join(bratumia_registered, '%s_Registered_.mha'%(modality_map[modality]))
        path_to_target = data_io.create_registered_image_path(subject=subject_id, session=session, modality=modality,
                                                        registration='rigid', other='withskull', extension='mha',
                                                              processing='bratumia')
        print("Copying '%s' -> '%s'"%(path_to_source, path_to_target))
        if copy:
            if (not os.path.exists(path_to_target)) or overwrite:
                shutil.copy(path_to_source, path_to_target)
            else:
                print("File '%s' already exists. Skipping...")
    #-- registered, skullstripped
    for modality in modalities:
        path_to_source = os.path.join(bratumia_registered_masked, '%s_Masked_.mha'%(modality_map[modality]))
        path_to_target = data_io.create_registered_image_path(subject=subject_id, session=session, modality=modality,
                                                        registration='rigid', other='skullstripped', extension='mha',
                                                              processing='bratumia')
        print("Copying '%s' -> '%s'" % (path_to_source, path_to_target))
        if copy:
            if (not os.path.exists(path_to_target)) or overwrite:
                shutil.copy(path_to_source, path_to_target)
            else:
                print("File '%s' already exists. Skipping...")
    #-- segmentations
    bratumia_seg_tumor = os.path.join(bratumia_registered_segmentation, "T1C_Tumor_%s_Registered.mha" % (folder))
    seg_tumor_target = data_io.create_registered_image_path(subject=subject_id, session=session, modality='tumorseg',
                                                    segmentation='tumor', other='bratumia', processing='bratumia',
                                                    extension='mha')
    print("Copying '%s' -> '%s'" % (bratumia_seg_tumor, seg_tumor_target))
    if copy:
        if (not os.path.exists(seg_tumor_target)) or overwrite:
            shutil.copy(bratumia_seg_tumor, seg_tumor_target)
        else:
            print("File '%s' already exists. Skipping...")

    bratumia_seg_all = os.path.join(bratumia_registered_segmentation, "T1C_HealthyAndTumor_%s_Registered.mha" % (folder))
    seg_all_target = data_io.create_registered_image_path(subject=subject_id, session=session, modality='tumorseg',
                                                        segmentation='all', other='bratumia', processing='bratumia',
                                                        extension='mha')
    print("Copying '%s' -> '%s'" % (bratumia_seg_all, seg_all_target))
    if copy:
        if (not os.path.exists(seg_all_target)) or overwrite:
            shutil.copy(bratumia_seg_all, seg_all_target)
        else:
            print("File '%s' already exists. Skipping...")


def convert_coord_to_dict(coord_tuple, prefix=''):
    coord_dict = {}
    for counter, value in enumerate(['x', 'y', 'z']):
        key = prefix+value
        coord_dict[key] = coord_tuple[counter]
    return coord_dict


def analyze_segmentations(data_io, subjects=None):
    if subjects is None:
        subjects = data_io.bids_layout.unique('subject')

    df = pd.DataFrame()
    for subject_id in subjects:
        sessions = data_io.bids_layout.get(target='session', subject=subject_id, return_type='id')
        for session in sessions:
            print("Analyzing segmentations for subject '%s', session '%s'"%(subject_id, session))

            curr_measures = {}
            curr_measures['subject_id'] = subject_id
            curr_measures['session'] = session

            for processing_mode in ['bratumia']:
                # == Compute segmentation label stats
                # -- Bratumia
                if processing_mode == 'bratumia':
                    path_to_segmentation_bratumia_all = data_io.create_registered_image_path(subject=subject_id,
                                                                                         session=session,
                                                                                         modality='tumorseg',
                                                                                         segmentation='all',
                                                                                         other='bratumia',
                                                                                         processing=processing_mode,
                                                                                         extension='mha',
                                                                                         create=False)

                    path_to_segmentation_bratumia_tumor = data_io.create_registered_image_path(subject=subject_id,
                                                                                             session=session,
                                                                                             modality='tumorseg',
                                                                                             segmentation='tumor',
                                                                                             other='bratumia',
                                                                                             processing=processing_mode,
                                                                                             extension='mha',
                                                                                             create=False)

                    # Give preference to files suffixed by '_p'
                    path_name = path_to_segmentation_bratumia_tumor.split('.')[0]
                    path_to_segmentation_prativa_tumor = path_name + '_p.mha'

                    if os.path.exists(path_to_segmentation_prativa_tumor):
                        path_to_seg_file = path_to_segmentation_prativa_tumor
                    elif os.path.exists(path_to_segmentation_bratumia_tumor):
                        path_to_seg_file = path_to_segmentation_bratumia_tumor

                    if os.path.exists(path_to_seg_file):
                        segmentation_img = sitk.ReadImage(path_to_seg_file)
                        seg_meas = ip.compute_volume(segmentation_img,
                                                     label_tissue_map=config.label_tissue_map_bratumia)
                        seg_meas_bratumia = {'bratumia_' + key: int(value) for key, value in
                                             seg_meas['volume_tissue'].items()}
                        curr_measures.update(seg_meas_bratumia)
                        curr_measures['segmentation_file'] = path_to_seg_file

            df = df.append(pd.DataFrame(curr_measures, index=[subject_id + '_' + session]))
    return df



def plot_segmentation_volumes(df, subject_ids = None,
                              plot_selection = ['Edema', 'EnhancingTumor'],
                              out_dir=None, show=True):
    # Remove 'bratumia_' from labels
    df.columns = [col.split("_")[-1] for col in df.columns]
    # Convert units: mm^3 -> cm^3=ml
    df = df[plot_selection]/1000.
    if subject_ids is None:
        subject_ids = df.reset_index().subject_id.unique()
    for subject_id in subject_ids:
        sel = df.loc[subject_id]
        # create plot
        fig, ax = plt.subplots(figsize=(6,4))
        sel.plot(kind='bar',ax=ax)
        #fig.subplots_adjust(right=0.7, bottom=0.2)
        # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper',
        #           ncol=1, borderaxespad=0.)
        leg = ax.legend(loc='upper center', frameon=True,
                  ncol=2, borderaxespad=0.5)
        leg.get_frame().set_linewidth(0.0)
        ax.set_ylabel("Volume [cm$^3$]")
        ax.set_xlabel("")
        plt.xticks(rotation=45)
        ax.set_title("Patient UPN %s"%subject_id)
        max_value = sel.max().max()
        ax.set_ylim(0, max_value + max_value*0.2)

        if out_dir is not None:
            gt.ensure_dir_exists(out_dir)
            save_name = "UPN-%s_segmentation_volumes.png"%(subject_id)
            fig.savefig(os.path.join(out_dir, save_name), bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
