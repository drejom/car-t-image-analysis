import os
import datetime
import config
from datetime import datetime


def get_file_extension(path_to_file):
    # need to test for existence of '.'
    # return None of component has no file extension
    file_name = os.path.split(path_to_file)[-1]
    file_name_split = file_name.split(".")
    if file_name_split[-1] == file_name:
        # there is no '.' in file_name
        return None
    else:
        return file_name_split[-1]


def ensure_dir_exists(path):
    if get_file_extension(path) == None:
        # assume that 'path' is directory, add trailing '/'
        path = path + '/'
    if os.path.exists(os.path.dirname(path)):
        return True
    else:
        os.makedirs(os.path.dirname(path))
        return False


def create_file_structure_path_names(subject_id, day, date, sequence, subject_prefix='', session_prefix='session_', base_path='.'):
    subject_dir_name = subject_prefix+subject_id
    session_dir_name = session_prefix+'%03d_%s' %(day, date.strftime('%Y-%m-%d'))
    acqu_dir_name    = sequence
    file_name = subject_dir_name + '_' + session_dir_name + '_' + acqu_dir_name
    path = os.path.join(base_path, subject_dir_name, session_dir_name, acqu_dir_name)
    return path, file_name


def parse_session_folder_name(name, session_prefix='session_'):
    if not name.startswith('.'):
        name = name[len(session_prefix):]
        name_split = name.split('_')
        day  =  int(name_split[0])
        date =  datetime.strptime(name_split[1], '%Y-%m-%d').date()
    else:
        day  = None
        date = None
    return day, date
#
# def parse_file_structure_path(path, subject_prefix='', session_prefix='session_'):
#     if os.path.isabs(path):
#         path = os.path.relpath(path, config.ivygap_dir_bids)
#
#     subpath, filename = os.path.split(path)
#     subpath_split = subpath.split('/')
#     print(subpath_split)
#     if len(subpath_split)==4:
#         _, subject_dir, session_dir, sequence_dir = subpath_split
#     elif len(subpath_split)==3:
#         subject_dir, session_dir, sequence_dir = subpath_split
#
#     day, date  = parse_session_folder_name(session_dir, session_prefix)
#     subject_id = subject_dir[len(subject_prefix):]
#     sequence   = session_dir
#
#     return subject_id, day, date, sequence
#
# datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
# path, filename = create_file_structure_path_names('W1', -2, datetime_object, 'seq1')
# subject_id, day, date, sequence = parse_file_structure_path(path)


def map_between_dicts(dict_1, dict_2):
    new_dict = {}
    for key_1, value_1 in dict_1.items():
        if key_1 in dict_2.keys():
            new_key = dict_2.get(key_1)
        else:
            new_key = key_1
        new_dict[new_key] = value_1
    return new_dict