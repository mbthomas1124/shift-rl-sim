declare -a StringArray=("155.246.104.89" 
                        "155.246.104.85"
                        "155.246.104.62" )


mode=$1

if [ "$mode" = "run" ]
then
	# Script of remote control functions 
	source ./automate.sh

    # Run experiments on remotes
    for val in ${StringArray[@]}; do
        run_ppo ${val} "~/shift-rl-sim/";
    done
elif [ "$mode" = "sync" ]
then
	# Script of remote control functions 
	source ./automate.sh

    # Kill python processes and SHIFT, then sync file
    for val in ${StringArray[@]}; do
        kill_all ${val};
        send_dir ${val} ~/shift-rl-sim/;
    done

elif [ "$mode" = "stop" ]
then
	# Script of remote control functions 
	source ./automate.sh

    # Kill python processes and SHIFT
    for val in ${StringArray[@]}; do
        kill_all ${val};
    done

elif [ "$mode" = "fetch" ]
then
	# Script of remote control functions 
	source ./automate.sh

    # Kill python processes and SHIFT, then sync file
    for val in ${StringArray[@]}; do
        fetch_dir ${val} "~/Results_Simulation/"
    done
else
	# Invalid command line argument for selected mode
	echo -e ${COLOR_ERROR}
	echo "Invalid mode. Please enter a valid mode."
	echo -e ${NO_COLOR}
fi
