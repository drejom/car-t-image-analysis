# COH Analysis

## Workflow for Segmentation & Analysis

Use scripts *coh_01*-*coh_04* in _analysis/irb_13384/scripts_:
1. *coh_01_**: 
    - parses dicom data from raw data folder
    - anonymizes data (removes information from header, replaces MRMs by UPNs)
    - identifies MRI sequences from header information

    Anonymized results are saved to path specified in `coh_config.coh_base_dir_out` (currently on server).
    The entire 'database' resulting from this analysis is saved locally to path specified in `coh_config.coh_path_to_metadata_raw_pkl`.
    
2. *coh_02_**:
    - allows manual inclusion of files that were not correctly recognized by (1).
    - copies all selected files to path specified in `config_coh.coh_dir_bids`.
    
3. *coh_03_**:
    - Identifies files to be segmented and organizes segmented files into folder structure.
    - Parts of the script need to be executed separately for the following steps:
      - Step 1:
        Generates input file(s) (csv) for automatic segmentation tool [BraTumIA (v2.0)](https://www.nitrc.org/projects/bratumia) to path specified in `config_coh.coh_dir_bratumia`.
      - Step 2:
        Manually run BraTumIA using the generated input files for batch processing.
      - Step 3:
        Copies resulting segmentations into directory structure located at `config_coh.coh_dir_bids`.
        
4. *coh_4_**:
    - Computes segmentation label statistics for all segmentations in `config_coh.coh_dir_bids`.
      Label files ending on '_p.mha' will be given preference over other existing segmentation label files for the same case.
      
         