#include "ZITrader.h"

#include <cmath>
#include <random>
#include <thread>

#include <shift/coreclient/FIXInitiator.h>

#include <shift/miscutils/statistics/BasicStatistics.h>

using namespace std::chrono_literals;

auto createRNG(const std::string& filename, bool load = false) -> std::mt19937
{
    std::mt19937 rng { std::random_device {}() };

    if (load) {
        std::ifstream inputFile { filename, std::ios::binary };
        if (inputFile) {
            inputFile >> rng;
        } else {
            load = false;
        }
    }

    if (!load) {
        std::ofstream outputFile { filename, std::ios::binary };
        if (outputFile) {
            outputFile << rng;
        }
    }

    return rng;
}

auto roundNearest(double value, double nearest) -> double
{
    return std::round(value / nearest) * nearest;
}

auto main(int argc, char** argv) -> int
{
    int clientNumber = atoi(argv[1]); // Client ID number
    std::string stockTicker { argv[2] }; // Stock ticker (e.g. CS1)
    double simulationDurationS = atof(argv[3]); // Duration of simulation (in seconds)
    double tradingRate = atof(argv[4]); // Number of trades per simulation session
    int confidenceLevel = atoi(argv[5]); // Confidence level (1, 2, or 3 - other means random)
    int riskAppetite = atoi(argv[6]); // Risk appetite (1, 2, or 3 - other means random)
    double initialPrice = atof(argv[7]); // Initial price
    double initialVolatility = atof(argv[8]); // Initial volatility (e.g 0.10)
    double minimumSpreadSize = atof(argv[9]); // Minimum spread size (e.g 0.01)
    bool repeatRandomSeed = (atoi(argv[10]) != 0); // Repeat random seed
    bool verboseMode = (atoi(argv[11]) != 0); // Verbose mode

    // For monetary amounts (std::put_money)
    std::cout.imbue(std::locale("en_US.UTF-8"));
    std::cout << std::showbase;
    // Avoid scientific notation in large double values
    std::cout << std::fixed << std::setprecision(2);

    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0')
        + std::to_string(clientNumber);

    int64_t simulationDurationUS = simulationDurationS * 1'000'000.0; // Seconds to microseconds

    double lastPrice = 0.0;
    double bestBid = 0.0;
    double bestAsk = 0.0;

    double targetPrice = 0.0;
    double targetRate = 0.0;
    double orderPrice = 0.0;
    int orderSize = 0;

    std::mt19937 gen = createRNG("rng/" + clientID, repeatRandomSeed);
    if (verboseMode) {
        std::cout << std::endl;
        if (!repeatRandomSeed) {
            std::cout << "INFO: RNG state saved to file!" << std::endl;
        } else {
            std::cout << "INFO: Loaded RNG state from file!" << std::endl;
        }
        std::cout << std::endl;
    }

    // https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    if (confidenceLevel < 1 || confidenceLevel > 3) {
        // P(X = x) = 1/3, for x in [0, 2]
        std::discrete_distribution<> confidenceLevelDD { 1.0, 1.0, 1.0 };
        confidenceLevel = confidenceLevelDD(gen) + 1; // Confidence level: 1, 2, or 3
    }

    if (riskAppetite < 1 || riskAppetite > 3) {
        // P(X = x) = 1/3, for x in [0, 2]
        std::discrete_distribution<> riskAppetiteDD { 1.0, 1.0, 1.0 };
        riskAppetite = riskAppetiteDD(gen) + 1; // Risk appetite: 1, 2, or 3
    }

    if (verboseMode) {
        std::cout << std::endl;
        std::cout << "Confidence Level: " << confidenceLevel << std::endl;
        std::cout << "Risk Appetite: " << riskAppetite << std::endl;
        std::cout << std::endl;
    }

    std::poisson_distribution<> tradingTimesPD { tradingRate };
    std::uniform_real_distribution<> tradingTimesURD { 0.0, 1.0 };
    // U(0.2, 0.4); or ,0.6); or ,0.8)
    std::uniform_real_distribution<> targetRateURD { 0.2, 0.2 + 0.2 * confidenceLevel };
    std::bernoulli_distribution orderSideBD { 0.5 };
    // N(0.0, 0.5); or ,1.0); or ,1.5)
    std::normal_distribution<> orderPriceND { 0.0, 0.5 * riskAppetite };

    int numberTrades = tradingTimesPD(gen);
    std::vector<int64_t> tradingTimesUS;

    tradingTimesUS.push_back(0LL);
    for (int i = 0; i < numberTrades; ++i) {
        tradingTimesUS.push_back(std::round(simulationDurationUS * tradingTimesURD(gen)));
    }
    tradingTimesUS.push_back(simulationDurationUS);

    // Sort trading times and, if duration of simulation is short, remove duplicates
    std::sort(tradingTimesUS.begin(), tradingTimesUS.end());
    tradingTimesUS.erase(
        std::unique(tradingTimesUS.begin(), tradingTimesUS.end()), tradingTimesUS.end());

    // Update num_trades taking begin and end of simulation into consideration
    numberTrades = tradingTimesUS.size();

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();

    ZITrader client { clientID, { stockTicker } };
    // std::string outputFile = "output/" + clientID + "_" + ZITrader::getLocalTime(true) + ".csv";
    // ZITrader client { clientID, (clientNumber % 20) == 0 ? outputFile : "", { stockTicker } };

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
    client.subOrderBook(stockTicker);

    // Trading strategy
    for (int i = 1; i < numberTrades; ++i) { // tradingTimes[0] == 0
        std::this_thread::sleep_for((tradingTimesUS[i] - tradingTimesUS[i - 1]) * 1us);

        if (verboseMode) {
            std::cout << std::endl;
            std::cout << std::setprecision(6);
            std::cout << "Trading Time: " << tradingTimesUS[i] / 1'000'000.0 << 's' << std::endl;
            std::cout << std::setprecision(2);
        }

        // Cancel last order if it has not executed yet
        if (client.getWaitingListSize() > 0) {
            if (verboseMode) {
                std::cout << "Canceling Pending Orders!" << std::endl;
            }
            client.cancelAllPendingOrders();
        }

        // Robot should not trade anymore
        if (i == (numberTrades - 1)) {
            break;
        }

        targetRate = targetRateURD(gen);
        if (verboseMode) {
            std::cout << "Target Rate: " << targetRate * 100.0 << '%' << std::endl;
        }

        // Required initial condition
        lastPrice = client.getLastPrice(stockTicker);
        lastPrice = (lastPrice > 0.0) ? lastPrice : initialPrice;

        if (orderSideBD(gen)) { // Limit Buy

            bestBid = client.getBestPrice(stockTicker).getBidPrice();
            bestBid = (bestBid > 0.0) ? bestBid : lastPrice;

            targetPrice = std::min(lastPrice, bestBid);

            if (verboseMode) {
                std::cout << "Last Price: " << std::put_money(lastPrice * 100.0) << std::endl;
                std::cout << "Best Bid: " << std::put_money(bestBid * 100.0) << std::endl;
                std::cout << "Target Price: " << std::put_money(targetPrice * 100.0) << std::endl;
            }

            orderPrice = roundNearest(
                targetPrice + initialVolatility * orderPriceND(gen), minimumSpreadSize);
            if (verboseMode) {
                std::cout << "Bid Price: " << std::put_money(orderPrice * 100.0) << std::endl;
            }

            orderSize = std::floor(
                (targetRate * client.getPortfolioSummary().getTotalBP()) / (100 * orderPrice));
            if (verboseMode) {
                std::cout << "Bid Size: " << orderSize << std::endl;
            }
            if (orderSize == 0) {
                if (verboseMode) {
                    std::cout << "Not submitting order of size: " << orderSize << std::endl;
                }
                continue;
            }

            shift::Order limitBuy { shift::Order::LIMIT_BUY, stockTicker, orderSize, orderPrice };
            client.submitOrder(limitBuy);

        } else { // Limit Sell

            bestAsk = client.getBestPrice(stockTicker).getAskPrice();
            bestAsk = (bestAsk > 0.0) ? bestAsk : lastPrice;

            targetPrice = std::max(lastPrice, bestAsk);

            if (verboseMode) {
                std::cout << "Last Price: " << std::put_money(lastPrice * 100.0) << std::endl;
                std::cout << "Best Ask: " << std::put_money(bestAsk * 100.0) << std::endl;
                std::cout << "Target Price: " << std::put_money(targetPrice * 100.0) << std::endl;
            }

            orderPrice = roundNearest(
                targetPrice + initialVolatility * orderPriceND(gen), minimumSpreadSize);
            if (verboseMode) {
                std::cout << "Ask Price: " << std::put_money(orderPrice * 100.0) << std::endl;
            }

            orderSize = std::floor(
                targetRate * (client.getPortfolioItem(stockTicker).getShares() / 100.0));
            if (verboseMode) {
                std::cout << "Ask Size: " << orderSize << std::endl;
            }
            if (orderSize == 0) {
                if (verboseMode) {
                    std::cout << "Not submitting order of size: " << orderSize << std::endl;
                }
                continue;
            }

            shift::Order limitSell { shift::Order::LIMIT_SELL, stockTicker, orderSize, orderPrice };
            client.submitOrder(limitSell);
        }

        if (verboseMode) {
            std::cout << std::endl;
        }
    }

    initiator.disconnectBrokerageCenter();

    return 0;
}
