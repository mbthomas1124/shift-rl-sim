#include <cmath>
#include <random>
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

auto brownianBridge(const std::vector<double>& values, const std::vector<int>& steps) -> std::vector<double>
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

auto main(int argc, char** argv) -> int
{
    std::string stockTicker { argv[1] }; // Stock ticker (e.g. XYZ)
    int simulationDuration = atoi(argv[2]); // Duration of simulation (in seconds)
    int samplingFrequency = atoi(argv[3]); // Sampling frequency (in seconds)
    double minimumDollarChange = atof(argv[4]); // Minimum dollar change (e.g 0.01)
    std::string outputDirectory { argv[5] }; // Output directory
    bool verboseMode = (atoi(argv[6]) != 0); // Verbose mode

    std::vector<double> prices = { 100.00, 105.00, 100.00, 95.00, 100.00, 105.00, 100.00 };
    std::vector<int> timePoints = { 0 * 60, 10 * 60, 20 * 60, 30 * 60, 40 * 60, 50 * 60, 60 * 60 };

    std::vector<double> path = brownianBridge(prices, timePoints);

    std::ostream* os;

    double lastPrice = 0.0;
    bool mustBuy = false;
    bool mustSell = false;

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();
    shift::CoreClient client { "marketmaker" };
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

    // Subscribe to order book data
    // client.subAllOrderBook();
    // client.subOrderBook(stockTicker);

    if (verboseMode) {
        os = &std::cout;
    } else {
        os = new std::ofstream(outputDirectory + "/" + stockTicker + "_" + getLocalTime(true) + ".csv");
    }

    *os << std::fixed << std::setprecision(2); // Price should always have to decimal places
    *os << "real_time,target_price,last_price,decision" << std::endl;

    while (client.getLastPrice(stockTicker) == 0.0) {
        std::this_thread::sleep_for(samplingFrequency * 1s);
    }

    std::chrono::microseconds elapsed;
    auto now = std::chrono::system_clock::now();
    auto begin = now;

    // Trading strategy
    for (int t = samplingFrequency; t <= simulationDuration; t += samplingFrequency) {
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1s - elapsed);

        now = std::chrono::system_clock::now();
        *os << getLocalTime() << ",";

        *os << path[t] << ",";

        lastPrice = client.getLastPrice(stockTicker);
        *os << lastPrice << ",";

        mustBuy = lastPrice < (path[t] - minimumDollarChange);
        mustSell = lastPrice > (path[t] + minimumDollarChange);

        if (mustBuy || mustSell) {
            if (client.getWaitingListSize() > 0) {
                client.cancelAllPendingOrders();
                *os << "CANCEL" << std::endl;
                continue;
            }
        }

        if (mustBuy) {
            shift::Order limitBuy(shift::Order::LIMIT_BUY, stockTicker, 100, path[t]);
            client.submitOrder(limitBuy);
            *os << "LIMIT_BUY" << std::endl;
        } else if (mustSell) {
            shift::Order limitSell(shift::Order::LIMIT_SELL, stockTicker, 100, path[t]);
            client.submitOrder(limitSell);
            *os << "LIMIT_SELL" << std::endl;
        } else {
            *os << "OK" << std::endl;
        }
    }

    return 0;
}
