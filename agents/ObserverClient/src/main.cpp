#include <map>
#include <thread>

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

auto main(int argc, char** argv) -> int
{
    int clientNumber = atoi(argv[1]); // Client ID number
    std::string stockTicker { argv[2] }; // Stock ticker (e.g. XYZ) - Use '?' to observe all tickers
    bool globalOrderBook = (atoi(argv[3]) != 0); // 0 = Local, 1 = Global order books
    int orderBookDepth = atoi(argv[4]); // Order book depth
    double simulationDurationS = atof(argv[5]); // Duration of simulation (in seconds)
    double samplingFrequencyS = atof(argv[6]); // Sampling frequency (in seconds)
    std::string outputDirectory { argv[7] }; // Output directory
    bool verboseMode = (atoi(argv[8]) != 0); // Verbose mode (overrides output directory)

    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0') + std::to_string(clientNumber);

    auto bidsBook = globalOrderBook ? shift::OrderBook::Type::GLOBAL_BID : shift::OrderBook::Type::LOCAL_BID;
    auto offersBook = globalOrderBook ? shift::OrderBook::Type::GLOBAL_ASK : shift::OrderBook::Type::LOCAL_ASK;

    int64_t simulationDurationUS = simulationDurationS * 1'000'000.0; // Seconds to microseconds
    int64_t samplingFrequencyUS = samplingFrequencyS * 1'000'000.0; // Seconds to microseconds

    std::map<std::string, std::ostream*> osMap; // (ticker, ostream*)

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();
    shift::CoreClient client { clientID };
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
    client.subAllOrderBook();

    if (stockTicker.front() == '?') { // Observe all tickers
        for (const std::string& ticker : client.getStockList()) {
            osMap[ticker] = nullptr;
        }
    } else { // Observe only specified ticker
        osMap[stockTicker] = nullptr;
    }

    if (verboseMode) {
        for (auto& kv : osMap) {
            kv.second = &std::cout;
        }
        std::cout << std::fixed << std::setprecision(2); // Price should always have two decimal places
        std::cout << "ticker,real_time,last_price";
        for (int i = 1; i <= orderBookDepth; ++i) {
            std::cout << ",bid_price_" << i << ",bid_size_" << i;
        }
        for (int i = 1; i <= orderBookDepth; ++i) {
            std::cout << ",ask_price_" << i << ",ask_size_" << i;
        }
        std::cout << std::endl;
    } else {
        for (auto& [ticker, ostreamPtr] : osMap) {
            ostreamPtr = new std::ofstream(outputDirectory + "/" + ticker + "_" + getLocalTime(true) + ".csv");
            *ostreamPtr << std::fixed << std::setprecision(2); // Price should always have two decimal places
            *ostreamPtr << "ticker,real_time,last_price";
            for (int i = 1; i <= orderBookDepth; ++i) {
                *ostreamPtr << ",bid_price_" << i << ",bid_size_" << i;
            }
            for (int i = 1; i <= orderBookDepth; ++i) {
                *ostreamPtr << ",ask_price_" << i << ",ask_size_" << i;
            }
            *ostreamPtr << std::endl;
        }
    }

    std::chrono::microseconds elapsed;
    auto now = std::chrono::system_clock::now();
    auto begin = now;

    for (int64_t t = samplingFrequencyUS; t <= simulationDurationUS; t += samplingFrequencyUS) {
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1us - elapsed);

        for (auto& [ticker, ostreamPtr] : osMap) {
            auto samplingTime = getLocalTime();
            auto lastPrice = client.getLastPrice(ticker);
            auto bids = client.getOrderBook(ticker, bidsBook, orderBookDepth);
            auto offers = client.getOrderBook(ticker, offersBook, orderBookDepth);

            *ostreamPtr << ticker;
            *ostreamPtr << ',' << samplingTime;
            *ostreamPtr << ',' << lastPrice;
            for (const auto& bid : bids) {
                *ostreamPtr << ',' << bid.getPrice() << ',' << bid.getSize();
            }
            for (int i = bids.size(); i < orderBookDepth; ++i) {
                *ostreamPtr << ",0.00,0";
            }
            for (const auto& offer : offers) {
                *ostreamPtr << ',' << offer.getPrice() << ',' << offer.getSize();
            }
            for (int i = offers.size(); i < orderBookDepth; ++i) {
                *ostreamPtr << ",0.00,0";
            }
            *ostreamPtr << std::endl;
        }
    }

    if (!verboseMode) {
        for (auto& kv : osMap) {
            dynamic_cast<std::ofstream*>(kv.second)->close();
            delete kv.second;
        }
    }

    initiator.disconnectBrokerageCenter();

    return 0;
}
