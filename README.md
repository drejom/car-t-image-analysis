# README
#
## Repository Structure

The repository contains general tools (*tools* folder) for the analysis of imaging studies,
as well as scripts and processing results of specific studies (*analysis* folder).

Frequently used data and selected processing results are stored as part
of the repository in the *data* folder.
Those (larger) files are managed using [git-lfs](https://git-lfs.github.com).

Various 'global' settings, such as local project paths, are defined in
*config.py* and can be made available throughout the project using
`import config`.


## How to use

### Get Data from git-lfs

After normal `git clone`, the *data* directory may still be empty.
In this case, perform 
```
git lfs pull
```

If [git-lfs](https://git-lfs.github.com) is not activated on your
system, follow installation instructions at (https://git-lfs.github.com).


### Install Dependencies

#### Only Jupyter Notebooks

If you are only interested in the final statistical analyses, without prior image processing, 
you can use the Jupyter notebooks in the relvant *analysis* subfolders.
These use processing results from the *data* folder.

The file *environment_notebook_analysis.yml* defines a minimum CONDA environment for
running the Jupyter notebooks included in this project.

**Steps for set-up:**
- Go to the root directory of the project *project-root*
- The command
    ```
    conda env create -f environment_notebook_analysis.yml
    ```
    will create a [CONDA](https://conda.io/docs/) environment with the name 'notebook-analysis'.
- Activate this environment by
    ```
    source activate notebook-analysis
    ```
- Start Jupyter notebook
    ```
    jupyter notebook
    ```
    Importing custom project libraries should work from within the ipython notebooks, 
    if jupyter has been started from *project-root*.
    Jupyter's working directory can be controlled by:
    ```
    jupyter notebook --notebook-dir=<path-to-dir>
    ```
      

#### Image Processing

For image processing more libraries are needed.
Major dependencies are:

- always:
    - pandas
    - matplotlib
    - numpy
- most image processing:
    - grabbit
    - SimpleITK
    - pydicom
    - dicom2nifti
    - vtk

If running [CONDA](https://conda.io/docs/) python on MacOS,
you can create a suitable python environment by
```
conda env create -f environment.yml
```

For image registration & LVd computation, the following additional dependencies need to be installed
and path / environment settings in `config.py` be updated for the specific local installation:  
    - [ANTs](http://stnava.github.io/ANTs/)

These dependencies can be installed using the `install_dependencies.sh` script.    
