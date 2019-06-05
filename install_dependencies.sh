
# Define installation directory

INST_DIR=/Users/mathoncuser/Desktop/DATA/car-t-image-analysis/ANTS

# Install ANTs from sources
# https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS
ANTS_FILES=${INST_DIR}/ants_git
ANT_INST_PATH=${INST_DIR}/ants_build
ANTS_PATH=${ANT_INST_PATH}/bin
echo "Installing ANTs..."
cd ${INST_DIR}
git clone https://github.com/stnava/ANTs.git ${ANTS_FILES}
mkdir ${ANT_INST_PATH}
cd ${ANT_INST_PATH}
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DSuperBuild_ANTS_USE_GIT_PROTOCOL=OFF -DRUN_LONG_TESTS=OFF ${ANTS_FILES}
make -j4
cp ${ANTS_FILES}/Scripts/* ${ANT_INST_PATH}/bin


# add environment variables to .bash_profile
BASH_FILE=~/.bash_profile
echo "Adding environment variables to ${BASH_FILE}"
echo "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2" >>${BASH_FILE}
echo "export ANTSPATH=${ANTS_PATH}" >>${BASH_FILE}
echo "export PATH=${ANTSPATH}:\$PATH" >>${BASH_FILE}
