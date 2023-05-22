#include "MACDTrader.h"

#include <cmath>
#include <thread>

#include <shift/coreclient/FIXInitiator.h>

using namespace std::chrono_literals;

auto getCurrentTime() -> std::string
{
    std::time_t currentTime = std::time(nullptr);
    char* currentTimeText = std::asctime(std::localtime(&currentTime));
    currentTimeText[std::strlen(currentTimeText) - 1] = 0;

    return std::string(currentTimeText);
}

auto roundNearest(double value, double nearest) -> double
{
    return std::round(value / nearest) * nearest;
}

auto simpleAverage(const std::vector<double>& values, size_t timePeriod = 0) -> double
{
    double result = 0.0;

    if (timePeriod == 0) {
        timePeriod = values.size();
    } else {
        timePeriod = (timePeriod <= values.size()) ? timePeriod : values.size();
    }

    for (size_t i = values.size() - timePeriod; i < values.size(); ++i) {
        result += values[i] / timePeriod;
    }

    return result;
}

auto standardDeviation(const std::vector<double>& values, double average, size_t timePeriod = 0) -> double
{
    double result = 0.0;

    if (timePeriod == 0) {
        timePeriod = values.size();
    } else {
        timePeriod = (timePeriod <= values.size()) ? timePeriod : values.size();
    }

    for (size_t i = values.size() - timePeriod; i < values.size(); ++i) {
        result += (values[i] - average) * (values[i] - average) / (static_cast<int>(timePeriod) - 1);
    }

    result = std::sqrt(result);

    return result;
}

auto exponentialMovingAverage(double previousAverage, double newValue, size_t timePeriod = 0) -> double
{
    double reductionFactor = 0.0;
    double result = 0.0;

    reductionFactor = 2.0 / (timePeriod + 1.0);
    result = (newValue - previousAverage) * reductionFactor + previousAverage;

    return result;
}

auto main(int argc, char** argv) -> int
{
    int clientNumber = atoi(argv[1]); // Client ID number
    std::string stockTicker { argv[2] }; // Stock ticker (e.g. XYZ)
    int simulationDuration = atoi(argv[3]); // Duration of simulation (in seconds)
    int samplingFrequency = atoi(argv[4]); // Sampling frequency (in seconds)
    size_t shortLag = atoi(argv[5]); // Lag for the short EMA (Exponential Moving Average)
    size_t longLag = atoi(argv[6]); // Lag for the long EMA (Exponential Moving Average)
    size_t signalLag = atoi(argv[7]); // Lag for the signal EMA (Exponential Moving Average)
    int positionSize = atoi(argv[8]); // Limit position size (1 size = 100 shares)
    double takeProfit = atof(argv[9]); // Take profit value (in dollars)
    double stopLoss = atof(argv[10]); // Stop loss value (in dollars)
    bool useMidPrice = (atoi(argv[11]) != 0); // Use mid price instead of last price
    bool useHistogram = (atoi(argv[12]) != 0); // Use histogram instead of change of signal
    bool useLimitOrders = (atoi(argv[13]) != 0); // Use limit orders instead of market orders
    bool verboseMode = (atoi(argv[14]) != 0); // Verbose mode
    bool fileOutput = (atoi(argv[15]) != 0); // Save output into a .csv file

    double meanPeakMultiplier = 1.0;
    if (argc > 16) {
        meanPeakMultiplier = atof(argv[16]);
    }

    // Avoid scientific notation in large double values
    std::cout << std::fixed << std::setprecision(2);

    // Add leading zeros to client id number
    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0') + std::to_string(clientNumber);

    // Reserve 5 minutes to clear all positions in the end
    simulationDuration -= 300;

    int elapsedTime = 0;

    std::vector<double> prices;
    double shortEMA = 0.0;
    double longEMA = 0.0;

    std::vector<double> macd;
    double signalEMA = 0.0;
    std::vector<double> histogram;

    bool positiveMovement = false;
    std::vector<double> peaks;
    double meanPeak = 0.0;

    int currentSignal = 0;
    int newSignal = 0;

    int orderSize = 0;
    double bidPrice = 0.0;
    double askPrice = 0.0;
    double takeProfitPrice = 0.0;
    double stopLossPrice = 0.0;

    std::ofstream ofs;
    std::string currentTime;

    if (fileOutput) {
        ofs.open(clientID + ".csv");
        ofs << std::fixed << std::setprecision(2);
        ofs << "Timestamp,Type,Value" << std::endl;
    }

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();
    MACDTrader client { clientID };
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

    // Wait until first trade of the day
    while (client.getLastPrice(stockTicker) == 0.0) {
        std::this_thread::sleep_for(samplingFrequency * 1s);
        elapsedTime += samplingFrequency;
    }

    // Wait until we have enough prices to compute the long EMA
    while (prices.size() < longLag) {
        std::this_thread::sleep_for(samplingFrequency * 1s);
        elapsedTime += samplingFrequency;

        if (useMidPrice) { // Use mid price
            prices.push_back(client.getMidPrice(stockTicker));
        } else { // Use last price
            prices.push_back(client.getLastPrice(stockTicker));
        }

        if (verboseMode) {
            std::cout << std::endl
                      << "Price: " << stockTicker << " " << prices[prices.size() - 1] << std::endl;
        }
    }

    shortEMA = simpleAverage(prices, shortLag);
    longEMA = simpleAverage(prices, longLag);

    // Wait until we have enough prices to compute the signal EMA
    while (macd.size() < signalLag) {
        std::this_thread::sleep_for(samplingFrequency * 1s);
        elapsedTime += samplingFrequency;

        if (useMidPrice) { // Use mid price
            prices.push_back(client.getMidPrice(stockTicker));
        } else { // Use last price
            prices.push_back(client.getLastPrice(stockTicker));
        }

        if (verboseMode) {
            std::cout << std::endl
                      << "Price: " << stockTicker << " " << prices[prices.size() - 1] << std::endl;
        }

        shortEMA = exponentialMovingAverage(shortEMA, prices[prices.size() - 1], shortLag);
        longEMA = exponentialMovingAverage(longEMA, prices[prices.size() - 1], longLag);
        macd.push_back(shortEMA - longEMA);
    }

    signalEMA = simpleAverage(macd, signalLag);
    histogram.push_back(macd[macd.size() - 1] - signalEMA);

    positiveMovement = (histogram[0] > 0);
    peaks.push_back(std::abs(histogram[0]));

    // Trading strategy
    for (int i = elapsedTime; i < simulationDuration; i += samplingFrequency) {
        std::this_thread::sleep_for(samplingFrequency * 1s);

        if (useMidPrice) { // Use mid price
            prices.push_back(client.getMidPrice(stockTicker));
        } else { // Use last price
            prices.push_back(client.getLastPrice(stockTicker));
        }

        if (verboseMode) {
            std::cout << std::endl
                      << "Price: " << stockTicker << " " << prices[prices.size() - 1] << std::endl;
        }

        shortEMA = exponentialMovingAverage(shortEMA, prices[prices.size() - 1], shortLag);
        longEMA = exponentialMovingAverage(longEMA, prices[prices.size() - 1], longLag);
        macd.push_back(shortEMA - longEMA);

        signalEMA = exponentialMovingAverage(signalEMA, macd[macd.size() - 1], signalLag);
        histogram.push_back(macd[macd.size() - 1] - signalEMA);

        if (positiveMovement) {
            if (histogram[histogram.size() - 1] > 0) {
                if (histogram[histogram.size() - 1] > peaks[peaks.size() - 1]) {
                    peaks[peaks.size() - 1] = histogram[histogram.size() - 1];
                }
            } else {
                positiveMovement = false;
                peaks.push_back(std::abs(histogram[histogram.size() - 1]));
            }
        } else {
            if (histogram[histogram.size() - 1] < 0) {
                if (std::abs(histogram[histogram.size() - 1]) > peaks[peaks.size() - 1]) {
                    peaks[peaks.size() - 1] = std::abs(histogram[histogram.size() - 1]);
                }
            } else {
                positiveMovement = true;
                peaks.push_back(histogram[histogram.size() - 1]);
            }
        }

        meanPeak = simpleAverage(peaks);

        if (useHistogram) {
            if (std::abs(histogram[histogram.size() - 1]) < peaks[peaks.size() - 1] && std::abs(histogram[histogram.size() - 1]) > meanPeak * meanPeakMultiplier) {
                if (macd[macd.size() - 1] < signalEMA) { // Still below signal line = buy signal
                    newSignal = 1;
                    if (client.getDiffShares(stockTicker) == 0 && currentSignal > 1) {
                        currentSignal = 1;
                    }
                } else if (macd[macd.size() - 1] > signalEMA) { // Still above signal line = sell signal
                    newSignal = -1;
                    if (client.getDiffShares(stockTicker) == 0 && currentSignal < -1) {
                        currentSignal = -1;
                    }
                }
            }
        } else {
            if (macd[macd.size() - 1] > signalEMA) { // Cross signal line from below = buy signal
                newSignal = 1;
            } else if (macd[macd.size() - 1] < signalEMA) { // Cross signal line from above = sell signal
                newSignal = -1;
            }
        }

        if (verboseMode) {
            std::cout << std::endl
                      << "Short EMA: " << shortEMA << std::endl
                      << "Long EMA: " << longEMA << std::endl
                      << "MACD: " << macd[macd.size() - 1] << std::endl
                      << "Signal EMA: " << signalEMA << std::endl
                      << "Histogram: " << histogram[histogram.size() - 1] << std::endl
                      << "Current Peak: " << std::abs(peaks[peaks.size() - 1]) << std::endl
                      << "Mean Peak: " << meanPeak << std::endl
                      << "New Signal: " << newSignal << std::endl
                      << "Current Signal: " << currentSignal << std::endl;
        }

        if (verboseMode) {
            if (client.getDiffShares(stockTicker) == 0 && client.getWaitingListSize() == 0) {
                std::cout << std::endl
                          << "Buying Power: " << client.getPortfolioSummary().getTotalBP() << std::endl;
            }
        }

        if (fileOutput) {
            currentTime = getCurrentTime();
            ofs << currentTime << ",Last Price," << client.getLastPrice(stockTicker) << std::endl;
            ofs << currentTime << ",Best Bid," << client.getBestBid(stockTicker).first << std::endl;
            ofs << currentTime << ",Best Ask," << client.getBestAsk(stockTicker).first << std::endl;
            ofs << std::fixed << std::setprecision(10);
            ofs << currentTime << ",Short EMA," << shortEMA << std::endl;
            ofs << currentTime << ",Long EMA," << longEMA << std::endl;
            ofs << currentTime << ",MACD," << macd[macd.size() - 1] << std::endl;
            ofs << currentTime << ",Signal EMA," << signalEMA << std::endl;
            ofs << currentTime << ",Histogram," << histogram[histogram.size() - 1] << std::endl;
            ofs << currentTime << ",Mean Peak," << meanPeak << std::endl;
            ofs << std::fixed << std::setprecision(2);

            if (client.getDiffShares(stockTicker) == 0 && client.getWaitingListSize() == 0) {
                ofs << currentTime << ",Buying Power," << client.getPortfolioSummary().getTotalBP() << std::endl;
            }
        }

        // Continue on buy signal
        if (newSignal == 1 && currentSignal >= 1) {

            if (client.getBestBid(stockTicker).first != bidPrice) {
                // Clear any non executed orders
                client.cancelAllPendingOrders();

                bidPrice = client.getBestBid(stockTicker).first;

                if (currentSignal >= 3) {
                    currentSignal = 2;
                }
            }

            if (client.getDiffShares(stockTicker) > 0) {
                if (orderSize == (client.getDiffShares(stockTicker) / 100) && currentSignal == 1) {
                    currentSignal = 2;
                }

                // Take Profit rule
                if (bidPrice >= takeProfitPrice && currentSignal != 3) {
                    currentSignal = 3;

                    if (verboseMode) {
                        std::cout << std::endl
                                  << "--- Take profit limit sell order!" << std::endl;
                    }

                    if (fileOutput) {
                        currentTime = getCurrentTime();
                        ofs << currentTime << ",Take Profit," << takeProfitPrice << std::endl;
                    }

                    if (useLimitOrders) {
                        client.createAndSubmitOrder(shift::Order::LIMIT_SELL, stockTicker, (client.getDiffShares(stockTicker) / 100), takeProfitPrice);
                    } else {
                        client.createAndSubmitOrder(shift::Order::MARKET_SELL, stockTicker, (client.getDiffShares(stockTicker) / 100));
                    }
                }
                // Stop Loss rule
                else if (bidPrice <= stopLossPrice && currentSignal != 4) {
                    currentSignal = 4;

                    if (verboseMode) {
                        std::cout << std::endl
                                  << "--- Stop loss limit sell order!" << std::endl;
                    }

                    if (fileOutput) {
                        currentTime = getCurrentTime();
                        ofs << currentTime << ",Stop Loss," << stopLossPrice << std::endl;
                    }

                    if (useLimitOrders) {
                        client.createAndSubmitOrder(shift::Order::LIMIT_SELL, stockTicker, (client.getDiffShares(stockTicker) / 100), stopLossPrice);
                    } else {
                        client.createAndSubmitOrder(shift::Order::MARKET_SELL, stockTicker, (client.getDiffShares(stockTicker) / 100));
                    }
                }
            }
        }

        // Continue on sell signal
        else if (newSignal == -1 && currentSignal <= -1) {

            if (client.getBestAsk(stockTicker).first != askPrice) {
                // Clear any non executed orders
                client.cancelAllPendingOrders();

                askPrice = client.getBestAsk(stockTicker).first;

                if (currentSignal <= -3) {
                    currentSignal = -2;
                }
            }

            if (client.getDiffShares(stockTicker) < 0) {
                if (orderSize == std::abs((client.getDiffShares(stockTicker) / 100)) && currentSignal == -1) {
                    currentSignal = -2;
                }

                // Take Profit rule
                if (askPrice <= takeProfitPrice && currentSignal != -3) {
                    currentSignal = -3;

                    if (verboseMode) {
                        std::cout << std::endl
                                  << "--- Take profit limit buy order!" << std::endl;
                    }

                    if (fileOutput) {
                        currentTime = getCurrentTime();
                        ofs << currentTime << ",Take Profit," << takeProfitPrice << std::endl;
                    }

                    if (useLimitOrders) {
                        client.createAndSubmitOrder(shift::Order::LIMIT_BUY, stockTicker, std::abs((client.getDiffShares(stockTicker) / 100)), takeProfitPrice);
                    } else {
                        client.createAndSubmitOrder(shift::Order::MARKET_BUY, stockTicker, std::abs((client.getDiffShares(stockTicker) / 100)));
                    }
                }
                // Stop Loss rule
                else if (askPrice >= stopLossPrice && currentSignal != -43) {
                    currentSignal = -4;

                    if (verboseMode) {
                        std::cout << std::endl
                                  << "--- Stop loss limit buy order!" << std::endl;
                    }

                    if (fileOutput) {
                        currentTime = getCurrentTime();
                        ofs << currentTime << ",Stop Loss," << stopLossPrice << std::endl;
                    }

                    if (useLimitOrders) {
                        client.createAndSubmitOrder(shift::Order::LIMIT_BUY, stockTicker, std::abs((client.getDiffShares(stockTicker) / 100)), stopLossPrice);
                    } else {
                        client.createAndSubmitOrder(shift::Order::MARKET_BUY, stockTicker, std::abs((client.getDiffShares(stockTicker) / 100)));
                    }
                }
            }
        }

        // Change signal
        else {
            // Clear any non executed orders
            client.cancelAllPendingOrders();
            currentSignal = newSignal;
        }

        // Initial buy signal
        if (currentSignal == 1 && client.getWaitingListSize() == 0) {

            if (verboseMode) {
                std::cout << std::endl
                          << "--- Buy signal!" << std::endl;
            }

            if (useLimitOrders) {
                bidPrice = client.getBestBid(stockTicker).first + 0.01;
            } else {
                bidPrice = client.getBestAsk(stockTicker).first;
            }

            takeProfitPrice = bidPrice + takeProfit;
            stopLossPrice = bidPrice - stopLoss;

            orderSize = positionSize;

            if (client.getDiffShares(stockTicker) != 0) {
                // If still did not close previous sell signal position, double dip!
                if (client.getDiffShares(stockTicker) < 0) {
                    orderSize += std::abs((client.getDiffShares(stockTicker) / 100));
                }
                // If still did not close previous buy signal position, hold on!
                else {
                    orderSize -= (client.getDiffShares(stockTicker) / 100);
                }
            }

            if (orderSize != 0) {

                // Check if funds are enough and account for slippage
                if (bidPrice * orderSize * 100 > client.getPortfolioSummary().getTotalBP()) {
                    orderSize = floor(client.getPortfolioSummary().getTotalBP() / (bidPrice * 1.1 * 100));
                }

                if (useLimitOrders) { // Signal Limit Buy
                    client.createAndSubmitOrder(shift::Order::LIMIT_BUY, stockTicker, orderSize, bidPrice);
                } else { // Signal Market Buy
                    client.createAndSubmitOrder(shift::Order::MARKET_BUY, stockTicker, orderSize);
                }

                if (orderSize > positionSize) { // Sanity check to correctly go from 1 to 2
                    orderSize = positionSize;
                }

                if (verboseMode) {
                    std::cout << std::endl
                              << "Take Profit Price:" << takeProfitPrice << std::endl
                              << "Stop Loss Price:" << stopLossPrice << std::endl;
                }

                if (fileOutput) {
                    currentTime = getCurrentTime();
                    ofs << currentTime << ",Buy Signal," << bidPrice << std::endl;
                }
            }
        }

        // Initial sell signal
        else if (currentSignal == -1 && client.getWaitingListSize() == 0) {

            if (verboseMode) {
                std::cout << std::endl
                          << "--- Sell signal!" << std::endl;
            }

            if (useLimitOrders) {
                askPrice = client.getBestAsk(stockTicker).first - 0.01;
            } else {
                askPrice = client.getBestBid(stockTicker).first;
            }

            takeProfitPrice = askPrice - takeProfit;
            stopLossPrice = askPrice + stopLoss;

            orderSize = positionSize;

            if (client.getDiffShares(stockTicker) != 0) {
                // If still did not close previous buy signal position, double dip!
                if (client.getDiffShares(stockTicker) > 0) {
                    orderSize += (client.getDiffShares(stockTicker) / 100);
                }
                // If still did not close previous sell signal position, hold on!
                else {
                    orderSize -= std::abs((client.getDiffShares(stockTicker) / 100));
                }
            }

            if (orderSize != 0) {

                // Check if enough stocks are available
                if (orderSize > (client.getPortfolioItem(stockTicker).getShares() / 100)) {
                    orderSize = (client.getPortfolioItem(stockTicker).getShares() / 100);
                }

                if (useLimitOrders) { // Signal Limit Sell
                    client.createAndSubmitOrder(shift::Order::LIMIT_SELL, stockTicker, orderSize, askPrice);
                } else { // Signal Market Sell
                    client.createAndSubmitOrder(shift::Order::MARKET_SELL, stockTicker, orderSize);
                }

                if (orderSize > positionSize) { // Sanity check to correctly go from -1 to -2
                    orderSize = positionSize;
                }

                if (verboseMode) {
                    std::cout << std::endl
                              << "Take Profit Price:" << takeProfitPrice << std::endl
                              << "Stop Loss Price:" << stopLossPrice << std::endl;
                }

                if (fileOutput) {
                    currentTime = getCurrentTime();
                    ofs << currentTime << ",Sell Signal," << askPrice << std::endl;
                }
            }
        }
    }

    // Cancel all pending orders
    client.cancelAllPendingOrders();
    // std::this_thread::sleep_for(15s);

    // Reset to initial position
    if (client.getDiffShares(stockTicker) > 0) { // Market Sell
        client.createAndSubmitOrder(shift::Order::MARKET_SELL, stockTicker, (client.getDiffShares(stockTicker) / 100));
    } else if (client.getDiffShares(stockTicker) < 0) { // Market Buy
        client.createAndSubmitOrder(shift::Order::MARKET_BUY, stockTicker, std::abs((client.getDiffShares(stockTicker) / 100)));
    }
    std::this_thread::sleep_for(15s);

    // Output final Buying Power
    if (verboseMode) {
        if (client.getDiffShares(stockTicker) == 0 && client.getWaitingListSize() == 0) {
            std::cout << std::endl
                      << "Buying Power: " << client.getPortfolioSummary().getTotalBP() << std::endl;
        }
    }

    // Output final Buying Power
    if (fileOutput) {
        currentTime = getCurrentTime();

        if (client.getDiffShares(stockTicker) == 0 && client.getWaitingListSize() == 0) {
            ofs << currentTime << ",Buying Power," << client.getPortfolioSummary().getTotalBP() << std::endl;
        }
    }

    initiator.disconnectBrokerageCenter();

    return 0;
}
