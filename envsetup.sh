#!/bin/bash

PYTHON_VER="3.9"
CURRENT_PLATFORM=$(uname -a)

CURRENT_PATH=$(pwd)
SRC_AREA=$(dirname $(realpath $BASH_SOURCE))

echo "Source directory: ${SRC_AREA}"
echo "Current directory: ${CURRENT_PATH}"

case $CURRENT_PLATFORM in
  *"cent7"* | *"sdf"* | *"el7"* | *"slc6"* | *"el9"*)
    case $CURRENT_PLATFORM in
      *"slc6"*)
        LCG_RELAESE="98bpython3"
        PLATFORM="x86_64-slc6-gcc8-opt"
      ;;

      *"el9"*)
        LCG_RELAESE="106"
        PLATFORM="x86_64-el9-gcc13-opt"
      ;;

      *)
        LCG_RELAESE="106"
        PLATFORM="x86_64-el9-gcc13-opt"
      ;;
    esac
    CVMFS_VIEWS=/cvmfs/sft.cern.ch/lcg/views
    setupATLAS -q
    export PIP_NO_CACHE_DIR=off
    lsetup "views LCG_${LCG_RELAESE} ${PLATFORM}"
    TEMP_PYTHONPATH=${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}/bin/python:${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}/lib
    TEMP_PYTHONPATH=${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}/bin/python:${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}/lib64:${TEMP_PYTHONPATH}
    TEMP_PYTHON_INCLUDE_PATH=${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}/bin/python:${CVMFS_VIEWS}/LCG_${LCG_RELAESE}/${PLATFORM}include/python${PYTHON_VER}
  ;;

  *)
    export PIP_NO_CACHE_DIR=off
  ;;
esac

OLD_PYTHONPATH=$PYTHONPATH
PYVENV_NAME=${1:-py3}

if [ -f "${SRC_AREA}/../${PYVENV_NAME}/bin/activate" ]; then
    source ${SRC_AREA}/../${PYVENV_NAME}/bin/activate
else
    python -m venv ${SRC_AREA}/../${PYVENV_NAME}
    source ${SRC_AREA}/../${PYVENV_NAME}/bin/activate
    python -m pip install -U pip setuptools wheel
    cd ${SRC_AREA}/
    echo 'Now run `python -m pip install -e .` or
        `python -m pip install -e .[unfolding]` or
        `python -m pip install -e .[complete]`'
    read -p "Press enter for basic install, or choose unfolding / complete: " answer
    case ${answer} in
        unfolding )
            python -m pip install -e .[unfolding]
        ;;
        complete )
            python -m pip install -e .[complete]
        ;;
        * )
            python -m pip install -e .
        ;;
    esac
    cd $CURRENT_PATH
fi
export PYTHONPATH=${SRC_AREA}/${PYVENV_NAME}/lib/python${PYTHON_VER}/site-packages:$OLD_PYTHONPATH:$TEMP_PYTHONPATH
export ROOT_INCLUDE_PATH=${SRC_AREA}/${PYVENV_NAME}/include:$ROOT_INCLUDE_PATH
export LD_LIBRARY_PATH=${SRC_AREA}/${PYVENV_NAME}/lib:$LD_LIBRARY_PATH:$TEMP_PYTHONPATH:/lib:/lib64
export PYTHON_INCLUDE_PATH=$TEMP_PYTHON_INCLUDE_PATH
