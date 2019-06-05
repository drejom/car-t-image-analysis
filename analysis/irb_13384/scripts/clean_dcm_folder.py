import pydicom
import os
import pandas as pd
import shutil


dcm_dir = '/Volumes/Macintosh HD-1/Users/mathoncuser/Desktop/DATA/CAR-T-CELL/XXXXXXXXXXXXX/AX.T1.MPR.post'

df = pd.DataFrame()
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(dcm_dir):
    print("Scanning directory '%s'. " %dirName)
    for filename in fileList:
        print("  - processing file '%s'." % filename)
        path_to_file = os.path.join(dirName, filename)
        try:
            ds = pydicom.dcmread(path_to_file, force=True)
            # extract info from dicom header
            attributes = ['study_description', 'study_date', 'series_description', 'patient_id', 'patient_name',
                          'series_instance_uid', 'study_instance_uid', 'patient_sex', 'patient_age',
                          'slice_thickness', 'spacing_between_slices', 'repetition_time', 'echo_time' ,
                          'inversion_time', 'mr_acquisition_type', 'sequence_variant', 'contrast_bolus_agent',
                          'protocol_name', 'series_number']

            meta_data = {}
            for attr in attributes:
                dcm_header_name = ''.join([comp.capitalize() for comp in attr.split('_')])
                if dcm_header_name.lower().endswith('id'): # correct issue with capitalisation of ID
                    dcm_header_name = dcm_header_name[:-2 ] +'ID'
                if hasattr(ds, dcm_header_name):
                    meta_data[attr] = getattr(ds, dcm_header_name)

            lstFilesDCM.append(os.path.join(dirName, filename))
            meta_data['path_to_dir'] = dirName
            meta_data['path_to_file'] = path_to_file
            series = pd.Series(meta_data)
        except:
            print("Could not process file '%s' " %path_to_file)

        if hasattr(series, 'patient_id'):
            if series.patient_id is not None:
                df = df.append(series, ignore_index=True)
            else:
                print("Did not find patient id, try again")

selection = df[df.series_number==21.0]

for file in selection.path_to_file.values:
    if os.path.exists(file):
        #os.remove(file.replace(' ', '\ '))
        os.remove(file)
    else:
        print("File '%s' does not exist."%file)