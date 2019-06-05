import pydicom
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import dicom2nifti
import shlex
import subprocess
import shutil
import config

import re
from tools import general_tools as gt



def create_pd_from_dcm_dir(dcm_dir, out_dir=None):
    files_to_exclude = ['.raw', '.png', '.txt', '.jpg', '.jpeg', 'xls', '.nii',
                        'bat', 'cbf', 'cbv', 'kep', 'ktr', 'leakage', 'r2_', 've', 'volume', 'vp', 'con_',
                        'mtt', 'tif']
    df = pd.DataFrame()
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dcm_dir):
        print("Scanning directory '%s'."%dirName)
        cnt_file = 0
        for filename in fileList:
            print("  - processing file '%s'." % filename)
            if cnt_file < 1:
                if not any(ext in filename.lower() for ext in files_to_exclude):
                    # since each folder represents 1 acquisition,
                    # we read only the first dicom file in the folder to get the folder metadata
                    path_to_file = os.path.join(dirName, filename)

                    try:
                        ds = pydicom.dcmread(path_to_file, force=True)

                        # extract info from dicom header
                        attributes = ['study_description', 'study_date', 'series_description', 'patient_id', 'patient_name',
                                      'series_instance_uid', 'study_instance_uid', 'patient_sex', 'patient_age',
                                      'slice_thickness', 'spacing_between_slices', 'repetition_time', 'echo_time' ,
                                      'inversion_time', 'mr_acquisition_type', 'sequence_variant', 'contrast_bolus_agent',
                                      'protocol_name']

                        meta_data = {}
                        for attr in attributes:
                            dcm_header_name = ''.join([comp.capitalize() for comp in attr.split('_')])
                            if dcm_header_name.lower().endswith('id'): # correct issue with capitalisation of ID
                                dcm_header_name = dcm_header_name[:-2]+'ID'
                            if hasattr(ds, dcm_header_name):
                                meta_data[attr] = getattr(ds, dcm_header_name)

                        lstFilesDCM.append(os.path.join(dirName, filename))
                        meta_data['path_to_dir'] = dirName
                        series = pd.Series(meta_data)
                    except:
                        print("Could not process file '%s'"%path_to_file)
                        cnt_file = cnt_file + 1

                    if hasattr(series, 'patient_id'):
                        if series.patient_id is not None:
                            df = df.append(series, ignore_index=True)
                            cnt_file = cnt_file + 1
                        else:
                            print("Did not find patient id, try again")


    # save
    if out_dir:
        gt.ensure_dir_exists(out_dir)
        df.to_excel(os.path.join(out_dir, 'data.xls'))
        df.to_pickle(os.path.join(out_dir, 'data.pkl'))
    return df

def get_items_in_string(items, string, return_which='all'):
    """
    This function searches a string for occurrences of the substrings in the 'items' variable
    and returns a list of the substrings found.

    'items' can be list or dictionary; if dictionary it is expected to have the form
    { new_name_A : [name_option_A1, name_option_A2, name_option_A3],
      new_name_B : [name_option_B1],
      etc
    }
    :param items:
    :param string:
    :return:
    """
    s = str(string).lower()
    if type(items)==dict:  # map to standard name
        # generated maps with current names as keys
        current_to_final_names_dict = {}
        for final_name, current_names in items.items():
            for current_name in current_names:
                current_to_final_names_dict[current_name] = final_name
        # Create list
        item_in_string = [current_to_final_names_dict[item] for item in current_to_final_names_dict.keys()
                          if item.lower() in s]
    else:
        item_in_string = [item for item in items if item.lower() in s]
    if len(item_in_string) > 0:
        if return_which=='all':
            return ' '.join(item_in_string)
        elif type(return_which)==int:
            return item_in_string[return_which]
        else:
            print("Do not understand value of 'return_which'")
    else:
        return False


def extract_with_regexp(string_to_parse, regexp):
    string_to_parse = str(string_to_parse)
    r = re.search(regexp, string_to_parse)
    extract = ''
    if not r == None:
        extract = string_to_parse[r.start():r.end()]
    return extract


def identify_sequences_bruker(df_in, series_descr_map, out_dir):

    df = df_in.copy(deep=True)
    # Expected sequences
    sequences          = ['T1w', 'T2w', 'T1wPost', 'T1Map', 'DiffusionMap', 'DCE', 'DCE_ref', 'DCE_FA']

    # Parse Series Description
    for name, value_map in series_descr_map.items():
        if name in ['keyword']:
            df[name] = df.protocol_name.apply(lambda x: get_items_in_string(value_map, x, return_which=-1))
        else:
            df[name] = df.protocol_name.apply(lambda x: get_items_in_string(value_map, x))
            if type(value_map) == dict:
                print(value_map)
                if list(value_map.keys())[0]=='num': # indicate regexp to extract numeric value
                    print(value_map)
                    df[name] = df.protocol_name.apply(lambda x: extract_with_regexp(x, value_map.get('num')))



    # Identify standard sequences based on parsing results, i.e. from components:
    # weighting  -- dimension -- sequence -- contrast

    # T1w = T1w & (not POST) & (not positioning)
    df['T1w'] = np.logical_and(
                    np.logical_and(df.weighting == 'T1w', np.logical_not(df.contrast == 'Post')),
                    np.logical_and(np.logical_not(df.keyword == 'positioning'),
                                   np.logical_not(df.keyword == 'map')))

    # T1w = T1w & POST
    df['T1wPost'] = np.logical_and(df.weighting == 'T1w', df.contrast == 'Post')

    # T2w = T2w
    df['T2w'] = df.weighting == 'T2w'

    # T1 MAP FA = T1w  & 'map'
    df['T1Map'] = np.logical_and(df.weighting == 'T1w', df.keyword == 'map')

    # Diffusion
    df['DiffusionMap'] = np.logical_and(df.sequence == 'SE', df.keyword == 'map')

    # DCE
    df['DCE'] = np.logical_and(df.keyword == 'DCE_dynamics', df.contrast == 'Post')
    df['DCE_ref'] = np.logical_and(df.keyword == 'DCE_dynamics', df.contrast == 'Pre')
    df['DCE_FA']  = df.keyword == 'DCE_FA'

    # Summarize available sequences per time points
    df['seqs_per_timepoint'] = df.groupby(['patient_id', 'study_instance_uid']).apply(lambda x:
                                                                                     [seq for seq in sequences if
                                                                                      any(x[seq])])
    for seq in sequences:
        df.loc[df[seq] == True, 'sequence_name'] = seq

    df.protocol_name = df.protocol_name.fillna('')

    # save
    df.to_excel(os.path.join(out_dir, 'data_with_sequences.xls'))
    df.to_pickle(os.path.join(out_dir, 'data_with_sequences.pkl'))

    return df




def identify_sequences(df_in, series_descr_map, out_dir=None):

    df = df_in.copy(deep=True)
    # Expected sequences
    sequences_anatomic = ['T1w','T1wPost', 'T2w', 'T2wFLAIR', 'T2wPD']
    sequences_dynamic  = ['DCE','DSC']
    sequences_map      = ['T1wMapFA']
    sequences_diffusion= ['DTI', 'DWI', 'ADC']
    sequences          = sequences_anatomic + sequences_diffusion + sequences_map + sequences_dynamic

    # Parse Series Description
    for name, value_map in series_descr_map.items():
        if name in ['keyword']:
            df[name] = df.series_description.apply(lambda x: get_items_in_string(value_map, x, return_which=-1))
            df[name+'_protocol'] = df.protocol_name.apply(lambda x: get_items_in_string(value_map, x, return_which=-1))
        else:
            df[name] = df.series_description.apply(lambda x: get_items_in_string(value_map, x))
            df[name+'_protocol'] = df.protocol_name.apply(lambda x: get_items_in_string(value_map, x))
            print(df[name])
            if type(value_map) == dict:
                print(value_map)
                if list(value_map.keys())[0]=='num': # indicate regexp to extract numeric value
                    print(value_map)
                    df[name] = df.series_description.apply(lambda x: extract_with_regexp(x, value_map.get('num')))



    # Identify standard sequences based on parsing results, i.e. from components:
    # weighting  -- dimension -- sequence -- contrast

    # T1w = T1w & (not POST) & (not 3D)
    df['T1w'] = np.logical_and(df.weighting == 'T1w', np.logical_and(
                                                      np.logical_not(df.contrast == 'Post'),
                                                      np.logical_not(df.keyword == 'SUB')))

    # T1w = T1w & POST & (not 3D)
    df['T1wPost'] = np.logical_and(df.weighting == 'T1w', np.logical_and(
                                                      df.contrast == 'Post',
                                                      np.logical_not(df.keyword == 'SUB')))

    # T2w = T2w & (not POST) & (not 3D) & (not FLAIR)
    df['T2w'] = np.logical_and(df.weighting == 'T2w',
                               np.logical_and(np.logical_not(df.sequence == 'FLAIR'),
                              np.logical_and(np.logical_not(df.contrast == 'Post'),np.logical_not(df.keyword == 'PD'))))

    # T2w = T2w & keyword = PD
    df['T2wPD'] = np.logical_and(df.weighting == 'T2w',df.keyword == 'PD')

    # T2w = T2w & (not POST) & (not 3D) & FLAIR
    df['T2wFLAIR'] = np.logical_and(df.weighting == 'T2w',
                                    np.logical_and(df.sequence == 'FLAIR',
                                                   np.logical_not(df.contrast == 'Post')))

    # T1 MAP FA = T1w  & MAP FA
    df['T1wMapFA'] = np.logical_and(df.weighting == 'T1w', df.sequence == 'MAP FA')

    # DCE = T1w & DCE
    df['DCE'] = df.keyword == 'DCE'

    # DSC = DSC
    df['DSC'] = df.keyword == 'DSC'

    # DTI = DTI
    df['DTI'] = df.keyword == 'DTI'

    # DWI = DWI
    df['DWI'] = df.keyword == 'DWI'

    # ADC = ADC
    df['ADC'] = df.keyword == 'ADC'

    # Summarize available sequences per time points

    # df['seqs_per_timepoint'] = df.groupby(['patient_id', 'study_instance_uid']).apply(lambda x:
    #                                                                                  [seq for seq in sequences if
    #                                                                                   any(x[seq])])
    #
    # df['anatomic_axial'] = df.groupby(['patient_id', 'study_instance_uid']).apply(lambda x:
    #                                                                              [seq for seq in sequences_anatomic if
    #                                                                               any(np.logical_and(x[seq],
    #                                                                                                  x.orientation == 'AX'))])

    for seq in sequences:
        df.loc[df[seq] == True, 'sequence_name'] = seq

    df.sequence_name = df.sequence_name.fillna('')



    # save
    if out_dir:
        df.to_excel(os.path.join(out_dir, 'data_with_sequences.xls'))
        df.to_pickle(os.path.join(out_dir, 'data_with_sequences.pkl'))

    return df


def plot_imaging_timeline(summary_df, out_dir, time_var='day'):
    plot_dict = collections.OrderedDict()
    plot_dict['T1w'] = 1.0
    plot_dict['T1w-post'] = 1.5
    plot_dict['T2w'] = 2.0
    plot_dict['T2-FLAIR'] = 2.5

    plot_dict['DCE'] = 3.5
    plot_dict['DSC'] = 4.0

    plot_dict['ADC'] = 5.0
    plot_dict['DWI'] = 5.5
    plot_dict['DTI'] = 6.0

    plot_dict_inv = {value: key for key, value in plot_dict.items()}

    for key, value in plot_dict.items():
        summary_df[key] = summary_df.seqs_per_timepoint.apply(lambda x: value if key in x else np.nan)

    for id in summary_df.index.levels[0]:
        print("-- plotting imaging timeline for patient '%s'"%id)
        fig, ax = plt.subplots()
        # summary_df.loc[id].plot('study_date',plot_dict.keys(), marker='x', linestyle='', ax=ax)
        summary_df.loc[id].plot(time_var, list(plot_dict.keys()), marker='x', linestyle='', ax=fig.gca())
        # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=3, mode="expand", borderaxespad=0.)
        ax.axvline(x=0)

        # labels = [ mpl.text.Text(0, ypos, label) for label, ypos in plot_dict.items()]
        # ax.set_yticklabels(labels)
        ax.set_yticks(list(plot_dict.values()))
        ax.set_yticklabels(list(plot_dict.keys()))
        ax.set_ylim(np.min(list(plot_dict.values())) - 0.5, np.max(list(plot_dict.values())) + 0.5)
        fig.savefig(os.path.join(out_dir, 'data_%s.png' % id))
        #plt.show()
        plt.clf()


def extract_dcm_header(dicom_dataset, path_to_file=None, anonymise=False):
    dcm_df = pd.DataFrame()
    dcm_tuples_to_remove = {
        "referring_physician" : ('0008','0090'),
        "physician_record"    : ('0008','1048'),
        "performing_phys_name"  : ('0008', '1050'),
        "operator_name"         : ('0008', '1070'),
        "patient_name"          : ('0010', '0010'),
        "patient_birth_date"    : ('0010', '0030'),
        "patient_address"       : ('0010', '1040'),
        "institution_name"      : ('0008', '0080'),
        "institution_address"   : ('0008', '0081'),
        "station_name"          : ('0008', '1010'),
        "requesting_physician"  : ('0032', '1032'),
        "requesting_service": ('0032', '1033'),
        "patient_id" : ('0010', '0020'),
        "patient_phone_number" : ('0010', '2154')
    }
    for de in dicom_dataset:
        #print(key, '-----', value)
        regexp_type = re.search('[A-Z]{2,2}:', str(de))
        dcm_info = { "index_1"      : str(de)[1:5],
                     "index_2"      : str(de)[7:11],
                     "index_name"   : str(de)[12:regexp_type.start()].strip(),
                     "type"         : str(de)[regexp_type.start():regexp_type.end()-1],
                     "info"         : str(de)[regexp_type.end():].strip()}

        if anonymise:
            for key, index_tuple in dcm_tuples_to_remove.items():
                index_1 = dcm_info['index_1']
                index_2 = dcm_info['index_2']
                if index_1==index_tuple[0] and index_2==index_tuple[1]:
                    dcm_info['info'] = '--- removed ---'

        # if hasattr(value,'keyword'):
        #     keyword = str(value.keyword)
        # else:
        #     keyword = ""
        #
        # if hasattr(value,'name'):
        #     name = str(value.name)
        # else:
        #     name = ""

        # keyword = str(value.keyword)
        # name = str(value.name)
        #
        # cur_value =""
        # if len(str(value.value)) < 1000:
        #     cur_value = str(value.value)
        #
        # if not keyword== 'PixelData':
        #     dcm_info = { "tag"           : str(value.tag),
        #                  "name"          : name,
        #                  "keyword"       : keyword,
        #                  "value"         : cur_value
        #                  }
        series = pd.Series(dcm_info)
        dcm_df = dcm_df.append(series, ignore_index=True)
        if not path_to_file==None:
            dcm_df.to_pickle(path_to_file + '.pkl')
            try:
                dcm_df.to_excel(path_to_file+'.xls')
                dcm_df.to_csv(path_to_file + '.csv')
            except:
                pass



def convert_t2pd_image(dicom_dir, tmp_output_dir, tmp_filename, target_path_t2, target_path_pd):
    """
    Splits interleaved T2 / PD image into 2 images, using https://github.com/rordenlab/dcm2niix.
    Requires dcm2niix to be installed, e.g. `brew install dcm2niix`
    """
    # file processing
    dcm2niix_cmd = "dcm2niix -d 0 -f %s -o %s %s" % (tmp_filename, tmp_output_dir, dicom_dir)
    args = shlex.split(dcm2niix_cmd)
    process = subprocess.Popen(args, env=config.path_environment)
    process.wait()
    print("dcm2niix_cmd terminated with return code: '%s'" % process.returncode)
    # file management
    if process.returncode==0:
        tmp_path_to_T2w = os.path.join(tmp_output_dir, tmp_filename + '_e2.nii')
        if os.path.exists(tmp_path_to_T2w):
            gt.ensure_dir_exists(target_path_t2)
            shutil.move(tmp_path_to_T2w, target_path_t2)
            print("- copied T2w image: %s -> %s"%(tmp_path_to_T2w, target_path_t2))
        tmp_path_to_PDw = os.path.join(tmp_output_dir, tmp_filename + '_e1.nii')
        if os.path.exists(tmp_path_to_PDw):
            gt.ensure_dir_exists(target_path_pd)
            shutil.move(tmp_path_to_PDw, target_path_pd)
            print("- copied T2w image: %s -> %s"%(tmp_path_to_PDw, target_path_pd))


def convert_dicom_to_nii(dicom_dir, output_dir, filename):
    """
    Assumes single output file
    Requires dcm2niix to be installed, e.g. `brew install dcm2niix`
    """
    # file processing
    output_dir = output_dir.replace(' ', '\ ')
    dicom_dir = dicom_dir.replace(' ', '\ ')
    dcm2niix_cmd = "dcm2niix -d 0 -s y -f %s -o %s %s" % (filename, output_dir, dicom_dir)
    args = shlex.split(dcm2niix_cmd)
    process = subprocess.Popen(args, env=config.path_environment)
    process.wait()
    print("dcm2niix_cmd terminated with return code: '%s'" % process.returncode)
    # expected file path
    path_to_nii_file = os.path.join(output_dir, filename + '.nii')
    path_to_json_file = os.path.join(output_dir, filename + '.json')
    if os.path.exists(path_to_json_file):
        os.remove(path_to_json_file)
    if os.path.exists(path_to_nii_file):
        success=True
    else:
        success=False
        print("Dicom dir '%s' was not converted"%dicom_dir)
    return success


def convert_dcm_folder(path_to_dcm_folder, path_to_output_folder, file_name,
                       export_dcm=False, anonymise=False, overwrite=False,
                       t2pd=False):
    gt.ensure_dir_exists(path_to_output_folder)
    path_to_out_file = os.path.join(path_to_output_folder, file_name)
    print("== Converting '%s'" % path_to_dcm_folder)
    # Get & Extract Metadata
    dcm_df = pd.DataFrame()
    if export_dcm:
        try:
            #print(path_to_out_file+'.csv')
            if not os.path.exists(path_to_out_file+'.csv') or overwrite:
                print("    -- Extracting dicom metadata")
                dcm_file = os.listdir(path_to_dcm_folder)[0]
                ds      = pydicom.dcmread(os.path.join(path_to_dcm_folder, dcm_file))
                dcm_df  = extract_dcm_header(ds, path_to_file=path_to_out_file, anonymise=anonymise)
        except:
            print("== Error extracting metadata for dicom folder '%s'" % (path_to_dcm_folder))
    # Convert dcm to nifty
    if not t2pd:
        print("    -- Converting dicom file to NIFTI")
        try:
            path_to_out_file = path_to_out_file+'.nii'
            if not os.path.exists(path_to_out_file) or overwrite:
                gt.ensure_dir_exists(os.path.dirname(path_to_out_file))
                #img_nii = dicom2nifti.dicom_series_to_nifti(path_to_dcm_folder, path_to_out_file)
                convert_dicom_to_nii(path_to_dcm_folder, path_to_output_folder, file_name)
        except:
            print("== Error converting dicom folder '%s'"%(path_to_dcm_folder))
    else:
        print("    -- Converting dicom file to NIFTI (T2wPD)")
        try:
            path_to_out_file = path_to_out_file+'.nii'
            if not os.path.exists(path_to_out_file) or overwrite:
                gt.ensure_dir_exists(os.path.dirname(path_to_out_file))
                target_path_t2 = os.path.join(path_to_output_folder, file_name + '.nii')
                pd_filename = "_".join(file_name.split("_")[:-1]) + "_PDfromT2wPD.nii"
                target_path_pd = os.path.join(path_to_output_folder, pd_filename)
                path_to_dicom_folder_escaped = path_to_dcm_folder.replace(' ', '\\ ')
                convert_t2pd_image(dicom_dir=path_to_dicom_folder_escaped,
                                   tmp_output_dir=path_to_output_folder,
                                   tmp_filename=file_name,
                                   target_path_t2=target_path_t2,
                                   target_path_pd=target_path_pd)
        except:
            print("== Error converting dicom folder '%s'"%(path_to_dcm_folder))

    return dcm_df

