import config
import os
import grabbit as gb
import json
import tools.general_tools as gt
from datetime import datetime
import shutil


def create_session_name(date, day):
    name="%s-day%03d" % (date.date().strftime('%Y-%m-%d'), day)
    return name

def get_string_with_suffix_from_list(string_list, suffix_list):
    strings_with_suffix = []
    for string in string_list:
        for suffix in suffix_list:
            if string.endswith(suffix):
                strings_with_suffix.append(string)
    return strings_with_suffix


#== PATH GENERATOR NO LONGER NEEDE -> using grabbit now -> DataIO
# class PathGenerator():
#     """
#     Consistent path generation throughout project.
#     Try to replace with path-pattern based generation from grabbit -> see function DataIO.copy_files
#     """
#     def __init__(self, base_path):
#         self.prefix_dict = {   'subject': 'sub-W',
#                                'session': 'ses-',
#                                'modality': '',
#                                'registered': 'reg',
#                                'segmentation': 'seg',
#                                'skullstripped': 'skullstripped'}
#         self.base_path = base_path
#
#     def generate_file_name_component(self, prefix_dict, text_dict, key):
#         prefix = prefix_dict.get(key)
#         text   = text_dict.get(key)
#         if prefix:
#             component = prefix+str(text)
#         else:
#             component = str(text)
#         return component
#
#
#     def generate_file_name(self, is_reg=False, is_seg=False, is_skullstripped=False, **kwargs):
#         file_name_components = []
#
#         #-- (1) NAME or SUBJECT_SESSION_MODALITY
#         if 'alternative_name' in kwargs.keys():
#             file_name_components.append(kwargs.get('alternative_name'))
#         elif 'subject' in kwargs.keys() and 'session' in kwargs.keys() and 'modality' in kwargs.keys():
#             file_name_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'subject'))
#             file_name_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'session'))
#             file_name_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'modality'))
#         else:
#             raise Exception
#
#         # -- (2) REGISTERED (optional)
#         if is_reg:
#             reg_component = self.prefix_dict.get('registered')
#             if 'reg_type' in kwargs.keys():
#                 reg_type = kwargs.get('reg_type')
#                 reg_component = reg_component+'-'+reg_type
#             if 'reg_to'  in kwargs.keys():
#                 reg_to = kwargs.get('reg_to')
#                 reg_component = reg_component + '-to-' + reg_to
#             file_name_components.append(reg_component)
#
#
#         # -- (3) SEGMENTATION (optional)
#         if is_seg:
#             seg_component = self.prefix_dict.get('segmentation')
#             if 'seg_descr' in kwargs.keys():
#                 seg_descr = kwargs.get('seg_descr')
#                 seg_component = seg_component + '-' + seg_descr
#             file_name_components.append(seg_component)
#
#
#         if is_skullstripped:
#             skullstr_component = self.prefix_dict.get('skullstripped')
#             file_name_components.append(skullstr_component)
#
#         file_name = "_".join(file_name_components)
#
#         #-- (4) EXTENSION
#         if 'extension' in kwargs.keys():
#             file_name = file_name+'.'+kwargs.get('extension')
#         return file_name
#
#
#     def generate_dir_name(self, abs_path=True, is_reg=False, is_skullstripped=False, **kwargs):
#         dir_components = []
#
#         #-- (1) NAME or SUBJECT_SESSION_MODALITY
#         if 'alternative_dir' in kwargs.keys():
#             dir_components.append(kwargs.get('alternative_name'))
#         elif 'subject' in kwargs.keys() and 'session' in kwargs.keys() and 'modality' in kwargs.keys():
#             dir_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'subject'))
#             if is_reg or is_skullstripped:
#                 dir_components.append('registered')
#             dir_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'session'))
#             dir_components.append(self.generate_file_name_component(self.prefix_dict, kwargs, 'modality'))
#         else:
#             raise Exception
#
#         path = os.path.join(*dir_components)
#         if abs_path:
#             path = os.path.join(self.base_path, path)
#         return path
#
#
#     def generate_path_name(self, is_reg=False, is_seg=False, abs_path=True, **kwargs):
#         file_name   = self.generate_file_name(is_reg=is_reg, is_seg=is_seg, **kwargs)
#         path        = self.generate_dir_name(abs_path=abs_path, is_reg=is_reg, **kwargs)
#         return os.path.join(path, file_name)
#
#     def generate_path(self, is_reg=False, is_seg=False, abs_path=True, **kwargs):
#         path = self.generate_path_name(is_reg=is_reg, is_seg=is_seg, abs_path=abs_path, **kwargs)
#         gt.ensure_dir_exists(os.path.dirname(path))
#         return path




class DataIO():

    def __init__(self, data_root, path_to_bids_config):
        self.path_to_bids_config    = path_to_bids_config
        #-- read file to extract path patterns
        with open(self.path_to_bids_config) as json_data:
            self.bids_config = json.load(json_data)
        #-- directory to which all other dirs are relative
        self.data_root              = data_root
        #-- initialize bids layout for reading
        self.init_bids_layout()
        #-- attach path generator instance
        #self.path_generator = PathGenerator(self.data_root)


    def init_bids_layout(self):
        self.bids_layout = gb.Layout([(self.data_root, self.path_to_bids_config)])
        #-- the attribute 'bids_layout.path_patterns' should be populated, but is not ...
        #   we do this manually:
        self.bids_layout.path_patterns = self.bids_config.get('default_path_patterns')


    def create_path(self, path_pattern_list=None, abs_path=True, create=True, **kwargs):
        if path_pattern_list:
            path = self.bids_layout.build_path(kwargs, path_pattern_list)
        else:
            path = self.bids_layout.build_path(kwargs)
        if abs_path:
            path = os.path.join(self.data_root, path)
        if create:
            gt.ensure_dir_exists(os.path.dirname(path))
        return path

    def create_image_path(self, subject, session, modality, abs_path=True, create=True, **kwargs):
        path = self.create_path(subject=subject, session=session, modality=modality,
                                processing='original', abs_path=abs_path, create=create, **kwargs)
        return path

    def create_registered_image_path(self, subject, session, modality, registration='rigid',
                                     abs_path=True, create=True, processing='registered', **kwargs):
        path = self.create_path(subject=subject, session=session, modality=modality, registration=registration,
                                processing=processing, abs_path=abs_path, create=create, **kwargs)
        return path
    #
    # def create_segmentation_image_path(self, subject, session, modality, segmentation,
    #                                    abs_path=True, create=True, processing='registered', **kwargs):
    #     path = self.create_path(subject=subject, session=session, modality=modality, segmentation=segmentation,
    #                             processing=processing, abs_path=abs_path, create=create, **kwargs)
    #     return path
    #
    # def create_registration_transform_path(self, subject, session, modality, registration='rigid', abs_path=True,
    #                                        create=True, processing='registered', **kwargs):
    #     path = self.create_path(subject=subject, session=session, modality=modality, registration=registration,
    #                             processing=processing, other='transform', abs_path=abs_path, create=create, **kwargs)
    #     return path

    # def create_registered_image_path_(self, subject, session, modality, registration='rigid', other='withskull',
    #                                  abs_path=True, create=True, processing='registered', **kwargs):
    #     path = self.create_path(subject=subject, session=session, modality=modality, registration=registration,
    #                             processing=processing, other=other, abs_path=abs_path, create=create, **kwargs)
    #     return path

    def create_segmentation_image_path(self, subject, session, modality, segmentation,
                                       abs_path=True, create=True, processing='registered', **kwargs):
        path = self.create_path(subject=subject, session=session, modality=modality, segmentation=segmentation,
                                processing=processing, abs_path=abs_path, create=create, **kwargs)
        return path

    def create_registration_transform_path(self, subject, session, modality, registration='rigid', abs_path=True,
                                           create=True, processing='registered', **kwargs):
        path = self.create_path(subject=subject, session=session, modality=modality, registration=registration,
                                processing=processing, other='transform', abs_path=abs_path, create=create, **kwargs)
        return path


    def get_image_files(self, img_type='any', name='any', extensions = ['nii', 'mha'], **kwargs):
        img_file_list = self.bids_layout.get(extensions=extensions, **kwargs)
        # == select for reg/no-reg
        if img_type == 'reg':
            file_list = [file for file in img_file_list if hasattr(file, 'registration')]
        elif img_type == 'noreg':
            file_list = [file for file in img_file_list if (not hasattr(file, 'registration'))]
        else:
            file_list = [file for file in img_file_list]

        if name == 'standard':
            file_list_2 = [file for file in file_list if (not hasattr(file, 'alternative_name'))]
        elif name== 'alternative':
            file_list_2 = [file for file in file_list if ( hasattr(file, 'alternative_name'))]
        else:
            file_list_2 = [file for file in file_list]

        file_list_3 = [file.filename for file in file_list_2]

        #img_file_list = get_string_with_suffix_from_list(seq_file_list, suffix_list=['nii', 'mha'])
        return file_list_3


    def get_reference_session_for_subject(self, subject):
        unique_sessions_subject = self.bids_layout.get(target='session', subject=subject, return_type='id')
        dates = [ datetime.strptime(session, '%Y-%m-%d').date() for session in unique_sessions_subject ]
        reference_date = sorted(dates)[0]
        return reference_date


    def get_reference_modality_for_session(self, subject, session, ref_seq_list=['T1w-3D', 'T1w', 'T1wPost-3D', 'T1wPost']):
        unique_modalities_subject_session = self.bids_layout.get(target='modality', subject=subject,
                                                                    session=session, return_type='id')
        ref_seq = None
        for seq in ref_seq_list:
            if seq in unique_modalities_subject_session:
                ref_seq = seq
                break
        return ref_seq


    def copy_files(self, new_base_dir='.', overwrite=False,
                   file_type='any', mode='copy', **kwargs ):
        """
        copies file in new directory tree
        Try to replace by bids_layout's copy_files function in the future.
        """
        #== retrieve query results
        file_list = self.bids_layout.get(**kwargs)
        #== select for reg/no-reg
        if file_type == 'reg':
            file_list = [file.filename for file in file_list if hasattr(file, 'registration')]
        elif file_type == 'noreg':
            file_list = [file.filename for file in file_list if (not hasattr(file, 'registration'))]
        else:
            file_list = [file.filename for file in file_list]

        if not file_type=='noreg':
            print("=== WARNING: directory structure for registered files is not handled correctly -- verify results!")

        #== generate new paths
        for old_path in file_list:
            new_path_rel = self.bids_layout.build_path(old_path, self.bids_layout.path_patterns)
            new_path_abs = os.path.join(new_base_dir, new_path_rel)
            gt.ensure_dir_exists(new_path_abs)
            if mode=='copy':
                print("Preparing to copy '%s' to '%s'"%(old_path, new_path_abs))
            elif mode=='move':
                print("Preparing to move '%s' to '%s'" % (old_path, new_path_abs))
            if os.path.exists(new_path_abs):
                if overwrite:
                    os.remove(new_path_abs)
                    shutil.copy(old_path, new_path_abs)
                else:
                    print("File '%s' already exists ... skipping."%(new_path_abs))
            else:
                shutil.copy(old_path, new_path_abs)
            if mode=='move':
                os.remove(old_path)
                try:
                    os.rmdir(old_path)
                except:
                    pass

        if not file_type=='noreg':
            print("=== WARNING: directory structure for registered files is not handled correctly -- verify results!")


#
# data = DataIO(project_root, config_file)
#
# file_name = data.create_image_path(subject='1', session='1232-23-23', modality='T1w', extension='mha')
# print(file_name)
#
# file_name = data.create_registered_image_path(subject='1', session='1232-23-23', modality='T1w', extension='mha', reg_type='affine')
# print(file_name)
#
# # file_name = generate_file_name(prefix_dict, subject='1', session='1232-23-23', modality='T1w', extension='mha',
# #                                is_reg=True, reg_type='affine', reg_to='some-other-image')
# # file_name = generate_file_name(prefix_dict, alternative_name='atlas', registered='affine', registered_to='someotherimage', extension='mha')
#
#
#
# pathgen = PathGenerator()
#
# path = pathgen.generate_path_name(subject='1', session='1232-23-23', modality='T1w', extension='mha')
