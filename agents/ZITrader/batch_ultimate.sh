
# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directories if they do not exist yet
[ -d output ] || mkdir output
[ -d rng ] || mkdir rng
function usage
{
    echo -e ${COLOR}
    echo "OVERVIEW: ZI Agents Set Up, Allows multile, Only works on CS1, CS2, CS3..."
    echo
    echo "USAGE: ./batch_ultimate.sh [options] <args>"
    echo
    echo "OPTIONS:";
    echo "  -h [ --help ]                        Display available options"
    echo "  -t [ --tickers] arg                  (int) declare number of tickers, [default]: 2"
    echo "  -a [ --agents] arg                   (int) declare number of agents, [default]: 100"
    echo "  -s [ --speed ] arg                   (int) declare the trading speed of the agents, [default]: 1"
    echo -e ${NO_COLOR}
}
#$start..$end}
function generate
{
    start=1
    end=${2}
    for i in $(seq 1 1 ${1})
    do

        for j in $(seq $start 1 $end)
        do
        echo $j
            nohup ./build/Release/ZITrader $j "CS$i" 23100.0 $((385*${3})) 2 2 100.00 0.10 0.01 0 0 </dev/null >/dev/null 2>&1 &
        done
        ((start+=${2}))
        ((end+=${2}))
    done

}
tickers=2
agents=100
speed=1

while [[ $# -gt 0 ]]; do
    case ${1} in
        -h | --help )
            usage
            exit 0
            ;;
        -t | --tickers )
            tickers=${2}
            echo ${2}
            #exit 0
            shift 2
            ;;
        -a | --agents )
            agents=${2}
            echo ${2}
            #exit 0
            shift 2
            ;;
        -s | --speed )
            speed=${2}
            echo ${2}
            #exit 0
            shift 2
            ;;
        *)
            echo
            echo -e "shift: ${COLOR_ERROR}error:${NO_COLOR} ${1} option is not available (please see usage with -h or --help)"
            echo
            exit 1
            ;;
    esac
done

generate $tickers $agents $speed

