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
BrokerageCenter -u acatsama -p cLbWKQzpWb9235NC -i SCATC_S2020 Acatsama hli113@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u coding_clowns -p A6QWxE3sEnsPQVAX -i SCATC_S2020 Coding_Clowns yshpileu@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u tbd -p KDJKdym8uf4zu7bL -i SCATC_S2020 TBD sbheda@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u seeking_alpha -p b6qx3dwJ3t9rK7ZW -i SCATC_S2020 Seeking_Alpha rjain18@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u two_and_twenty_llp -p nXxS3qxkdLUTkkF9 -i SCATC_S2020 Two_and_Twenty_LLP jlawrenc@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u ovacho -p v94uEpG6Y66dvyBF -i SCATC_S2020 Ovacho mzebrows@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u twsa -p Ws4JpbUd6hNwurHu -i SCATC_S2020 TWSA amaza@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u go_beyond -p G4Ug8M97xEUjyJ3J -i SCATC_S2020 Go_Beyond lzhen@stevens.edu  -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u alpha_seeker -p pLdngGMs5w4fmtRD -i SCATC_S2020 Alpha_Seeker jlu30@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u currency -p f72jtNkEpaJAYBtL -i SCATC_S2020 Currency ssaharka@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
BrokerageCenter -u team-110 -p t7TAFYjhbpN2qSsV -i SCATC_S2020 Team-110 aalakoc@stevens.edu -s
ERROR_CODES=$((${ERROR_CODES} + ${?}))
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
