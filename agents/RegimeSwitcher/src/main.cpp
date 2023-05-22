#include <boost/log/trivial.hpp>
#include <cmath>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <thread>
#include <vector>

#include <shift/coreclient/CoreClient.h>
#include <shift/coreclient/FIXInitiator.h>

using namespace std::chrono_literals;

auto getLocalTime(bool underscore = false) -> std::string
{
    std::ostringstream oss;

    auto now = std::chrono::system_clock::now();
    std::time_t now_t = std::chrono::system_clock::to_time_t(now); // loses microseconds information
    std::string us = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now - std::chrono::system_clock::from_time_t(now_t)).count());

    if (underscore) {
        oss << std::put_time(std::localtime(&now_t), "%F_%T");
    } else {
        oss << std::put_time(std::localtime(&now_t), "%F %T");
    }
    oss << '.' << std::string(6 - us.length(), '0') + us; // microseconds with "0" padding

    return oss.str();
}

auto roundNearest(double value, double nearest) -> double
{
    return std::round(value / nearest) * nearest;
}

auto getJumpIntensities(const std::vector<double>& values, const std::vector<int>& steps) -> std::vector<double>
{
    if (values.size() != steps.size()) {
        return std::vector<double> {};
    }

    std::vector<double> path;

    std::mt19937 gen { std::random_device {}() };
    std::normal_distribution<> d { 0.0, 1.0 };
    double dt = 0.0;
    double sqrtDT = 0.0;
    double t = 0.0;
    double previous = 0.0;
    double next = 0.0;

    path.push_back(values[0]);

    for (unsigned int i = 1; i < values.size(); ++i) {
        dt = 1.0 / (steps[i] - steps[i - 1]);
        sqrtDT = std::sqrt(dt);

        for (int j = 0; j < (steps[i] - steps[i - 1]); ++j) {
            t = j * dt;
            previous = path[path.size() - 1];
            next = previous + (values[i] - previous) * dt / (1 - t) + sqrtDT * d(gen);
            path.push_back(roundNearest(next, 0.01));
        }
    }

    return path;
}

/*

There needs to be 3 variables 
Inter arrival times in exponential dist
Intensity of the jump from a Uniform distribution

interarrival times Xi are exponential RVs with rate λ
exponential pdf f(x) = λ exp(−λx)

The arrival time of the jump is r.v from a Poisson process
http://www.math.wsu.edu/faculty/genz/416/lect/l05-45.pdf


*/
auto main(int argc, char** argv) -> int
{
    /*
    BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
    BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
    BOOST_LOG_TRIVIAL(info) << "An informational severity message";
    BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
    BOOST_LOG_TRIVIAL(error) << "An error severity message";
    BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";
    */

    std::string stockTicker { argv[1] }; // Stock ticker (e.g. XYZ)

    /*
    int simulationDuration = atoi(argv[2]); // Duration of simulation (in seconds)
    int samplingFrequency = atoi(argv[3]); // Sampling frequency (in seconds)
    double minimumDollarChange = atof(argv[4]); // Minimum dollar change (e.g 0.01)
    */
    std::string outputDirectory { argv[5] }; // Output directory
    bool verboseMode = (atoi(argv[6]) != 0); // Verbose mode

    //Log Set up the Logging
    std::ostream* os;
    if (verboseMode) {
        os = &std::cout;
    } else {
        os = new std::ofstream(outputDirectory + "/" + stockTicker + "_" + getLocalTime(true) + ".csv");
    }

    *os << std::fixed << std::setprecision(2); // Price should always have to decimal places
    *os << "real_time,target_price,last_price,decision" << std::endl;

    std::cout << "Log set up complete proceeding to SHIFT Connection Set up" << std::endl;

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();
    shift::CoreClient client { "regimeswitcher" };
    client.setVerbose(verboseMode);
    try {
        initiator.connectBrokerageCenter("initiator.cfg", &client, "password", verboseMode);
    } catch (const std::exception& e) {
        if (verboseMode) {
            std::cout << std::endl;
            std::cout << "Something went wrong: " << e.what() << std::endl;
            std::cout << std::endl;
        }
        return 1;
    }

    std::cout << "SHIFT Connection Set up compelte" << std::endl;

    // seed the RNG
    std::random_device rd; // uniformly-distributed integer random number generator
    std::mt19937 rng(rd()); // mt19937: Pseudo-random number generation

    // Inter Arrival times
    double averageArrival = 15;
    double lambda = 1 / averageArrival;
    std::exponential_distribution<double> exp(lambda);

    double sumArrivalTimes = 0;
    double newArrivalTime = 0;

    int numberOfJumps = 10;
    std::vector<int> arrivalTime(numberOfJumps);

    for (int i = 0; i < numberOfJumps; ++i) {
        newArrivalTime = exp(rng); // generates the next random number in the distribution
        sumArrivalTimes = sumArrivalTimes + newArrivalTime;
        std::cout << "newArrivalTime:  " << newArrivalTime << "    ,sumArrivalTimes:  " << sumArrivalTimes << std::endl;
        arrivalTime[i] = int(ceil(sumArrivalTimes));
    }

    std::cout << "Arrival times created" << std::endl;

    // Jump Intensities
    double sumJumpSize = 0;

    std::uniform_real_distribution<> jumpSizeUD { 0.0, 1.0 };
    std::vector<double> jumpSize(numberOfJumps);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < numberOfJumps; ++i) {
        double currentJumpSize = jumpSizeUD(rng);
        sumJumpSize = sumJumpSize + currentJumpSize;
        std::cout << "next jump Size:  " << currentJumpSize << "    ,sumJumpSize:  " << sumJumpSize << std::endl;
        jumpSize[i] = sumJumpSize;
    }

    std::cout << "Jump sizes created" << std::endl;

    std::vector<int> timePoints = { arrivalTime[1] * 60, arrivalTime[2] * 60, arrivalTime[3] * 60, arrivalTime[4] * 60, arrivalTime[5] * 60, arrivalTime[6] * 60, arrivalTime[7] * 60 };

    for (int i = 0; i <= numberOfJumps; i++) {

        /*
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1s - elapsed);

        now = std::chrono::system_clock::now();
        *os << getLocalTime() << ",";
        */

        //sleep till time W[1] is reached
        std::cout << "SLEEP TILL SECONDS : " << arrivalTime[i] << std::endl;
        std::this_thread::sleep_for(arrivalTime[i] * 1s);

        std::cout << "TIME TO CHANGE REGIME  : SLEPT FOR " << arrivalTime[i] << std::endl;

        double lastPrice = 0.0;
        lastPrice = client.getLastPrice(stockTicker);
        std::cout << "TIME TO CHANGE REGIME  : SLEPT FOR " << arrivalTime[i] << std::endl;
        std::cout << "TARGET PRICE IS : " << lastPrice + (lastPrice * jumpSize[i]) << " CURRENT PRICE IS :" << lastPrice << std::endl;

        /*
        if (client.getWaitingListSize() > 0) {
            client.cancelAllPendingOrders();
            *os << "CANCEL" << std::endl;
        }
        */

        int timeBtTrades = 2;

        for (int t = 0; t <= 10; t++) {
            // Trade till the target price is reached

            shift::Order limitBuy(shift::Order::LIMIT_BUY, stockTicker, 500, lastPrice + (lastPrice * jumpSize[i]));
            client.submitOrder(limitBuy);
            std::cout << "LIMIT_BUY" << std::endl;

            std::this_thread::sleep_for(timeBtTrades * 1s);

            //TODO: need to write code to sell aswell here

            //while (client.getLastPrice(stockTicker) == 0.0) {
        }
    }
    return 0;
}
