#include <deque>
#include <iostream>
#include <thread>
#include <vector>

#include <shift/coreclient/CoreClient.h>
#include <shift/coreclient/FIXInitiator.h>

#include <shift/miscutils/statistics/BasicStatistics.h>

using namespace std::chrono_literals;

auto main(int argc, char** argv) -> int
{
    int clientNumber = 0;
    std::vector<std::string> stockTickers = { "AAPL", "CSCO", "INTC", "MSFT" };
    int64_t simulationDuration = 300LL; // 300s = 5min
    double samplingFrequency = 1.0; // 1s
    unsigned int samplingWindow = 30;
    bool useMidPrice = false;

    // std::cout << std::fixed << std::setprecision(2); // Avoid scientific notation in large double values

    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0') + std::to_string(clientNumber);

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();
    shift::CoreClient client { clientID };
    client.setVerbose(true);
    try {
        initiator.connectBrokerageCenter("initiator.cfg", &client, "password", true);
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "Something went wrong: " << e.what() << std::endl;
        std::cout << std::endl;
        return 1;
    }

    // Subscribe to order book data
    client.subAllOrderBook();

    auto startTime = std::chrono::system_clock::now();

    client.requestSamplePrices(stockTickers, samplingFrequency, samplingWindow + 1);

    while (client.getLogReturnsSize(stockTickers[0]) < samplingWindow) {
        std::cout << "# Log Returns: " << client.getLogReturnsSize(stockTickers[0]) << std::endl;
        std::this_thread::sleep_for(samplingFrequency * 1s);
    }

    std::cout << std::endl;

    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - startTime).count() < simulationDuration) {
        for (const std::string& ticker : stockTickers) {
            auto logReturns = client.getLogReturns(ticker, useMidPrice);
            auto logReturns0 = client.getLogReturns(stockTickers[0], useMidPrice);

            std::cout << ticker << ':' << std::endl;
            std::cout << "Mean: " << shift::statistics::mean(logReturns) << std::endl;
            std::cout << "Variance: " << shift::statistics::variance(logReturns) << std::endl;
            std::cout << "Standard Deviation: " << shift::statistics::stddev(logReturns) << std::endl;
            std::cout << "Covariance (" << stockTickers[0] << "): "
                      << shift::statistics::covariance(logReturns, logReturns0) << std::endl;
            std::cout << "Correlation (" << stockTickers[0] << "): "
                      << shift::statistics::correlation(logReturns, logReturns0) << std::endl;
        }
        std::cout << std::endl;
        std::this_thread::sleep_for(samplingFrequency * 1s);
    }

    initiator.disconnectBrokerageCenter();

    return 0;
}
