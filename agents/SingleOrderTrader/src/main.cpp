#include <cmath>
#include <thread>

#include <shift/coreclient/CoreClient.h>
#include <shift/coreclient/FIXInitiator.h>

using namespace std::chrono_literals;

auto main(int argc, char** argv) -> int
{
    int clientNumber = atoi(argv[1]); // Client ID number
    std::string stockTicker { argv[2] }; // Stock ticker (e.g. XYZ)
    double orderTimeS = atof(argv[3]); // Order time (in seconds)
    int orderSide = atoi(argv[4]); // 1 : buy | -1 : sell
    double targetRate = atof(argv[5]); // Rate in which to buy or sell: (0.0, 1.0)
    bool verboseMode = (atoi(argv[6]) != 0); // Verbose mode

    std::cout << std::fixed << std::setprecision(2); // Avoid scientific notation in large double values

    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0') + std::to_string(clientNumber);

    int64_t orderTimeUS = orderTimeS * 1'000'000.0; // Seconds to microseconds

    double orderPrice = 0.0;
    int orderSize = 0;

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
    // client.subAllOrderBook();
    client.subOrderBook(stockTicker);

    // Trading strategy
    std::this_thread::sleep_for(orderTimeUS * 1us);

    if (orderSide > 0) { // Market Buy

        if (verboseMode) {
            std::cout << "--- Buy Signal!" << std::endl;
        }

        orderPrice = client.getLastPrice(stockTicker);
        orderSize = std::floor((targetRate * client.getPortfolioSummary().getTotalBP()) / (100 * orderPrice));

        shift::Order marketBuy { shift::Order::MARKET_BUY, stockTicker, orderSize };
        client.submitOrder(marketBuy);

    } else if (orderSide < 0) { // Market Sell

        if (verboseMode) {
            std::cout << "--- Sell Signal!" << std::endl;
        }

        orderSize = std::floor(targetRate * (client.getPortfolioItem(stockTicker).getShares() / 100.0));

        shift::Order marketSell { shift::Order::MARKET_SELL, stockTicker, orderSize };
        client.submitOrder(marketSell);
    }

    // Post trading
    std::this_thread::sleep_for(1min);

    // Cancel last order if it has not executed yet
    if (client.getWaitingListSize() > 0) {
        if (verboseMode) {
            std::cout << "Canceling Pending Orders!" << std::endl;
        }
        client.cancelAllPendingOrders();
    }

    initiator.disconnectBrokerageCenter();

    return 0;
}
