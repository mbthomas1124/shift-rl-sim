#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Incorrect number of arguments supplied"
    exit
fi

# number of cores of CPU
CORE_NUM=1
if [ "$(uname)" == "Linux" ]; then
	if [ ! $(nproc) -eq 1 ]
	then
        CORE_NUM=$(($(nproc) - 1))
    fi
fi
if [ "$(uname)" == "Darwin" ]; then
	if [ ! $(sysctl -n hw.physicalcpu) -eq 1 ]
	then
		CORE_NUM=$(($(sysctl -n hw.physicalcpu) - 1))
	fi
fi

# delete previous build directory
[ -d ${1}/build ] && rm -r ${1}/build

# build & install
cmake -H${1} -B${1}/build -DCMAKE_BUILD_TYPE=Release
cmake --build ${1}/build -- -j${CORE_NUM}
