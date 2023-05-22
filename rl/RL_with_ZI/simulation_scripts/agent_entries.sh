#!/bin/bash

# different output colors
COLOR='\033[1;36m'          # bold;cyan
COLOR_WARNING='\033[1;33m'  # bold;yellow
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'

# sum of error codes returned by the BrokerageCenter: 0 means all users were successfully created
ERROR_CODES=0

###############################################################################
BrokerageCenter -u marketmaker -p password -i Market Maker marketmaker@shift -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
###############################################################################

###############################################################################
for i in $(seq -f "%03g" 0 220)
do
    BrokerageCenter -u agent$i -p password -i Agent $i agent@shift -s
    ERROR_CODES=$((${ERROR_CODES} + ${?}))
done
###############################################################################

if [ ${ERROR_CODES} -eq 0 ]
then
    echo
    echo -e "status: ${COLOR_PROMPT}all users were successfully created${NO_COLOR}"
    echo
    exit 0
else
    echo
    echo -e "status: ${COLOR_ERROR}some of the users failed to be created${NO_COLOR}"
    echo
    exit 1
fi
