#include "RhoTrader.h"

#include <cmath>
#include <random>
#include <thread>

#include <shift/coreclient/FIXInitiator.h>

#include <shift/miscutils/statistics/BasicStatistics.h>

using namespace std::chrono_literals;

auto simpleAverage(const std::deque<double>& values, size_t timePeriod = 0) -> double
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

auto standardDeviation(const std::deque<double>& values, double average, size_t timePeriod = 0) -> double
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

/* 
 * https://en.wikipedia.org/wiki/Simple_linear_regression
 * "This shows that rho is the slope of the regression line of the standardized data points (and that this line passes through the origin)."
 */
auto main(int argc, char** argv) -> int
{
    int clientNumber = atoi(argv[1]); // Client ID number
    std::string stockTickerY { argv[2] }; // Stock ticker Y (e.g. CS1)
    std::string stockTickerX { argv[3] }; // Stock ticker X (e.g. CS2)
    double simulationDurationS = atof(argv[4]); // Duration of simulation (in seconds)
    double samplingFrequencyS = atof(argv[5]); // Sampling frequency (in seconds)
    int samplingWindow = atoi(argv[6]); // Sampling window
    int positionSizeY = atoi(argv[7]); // Position size Y (1 size = 100 shares)
    int positionSizeX = atoi(argv[8]); // Position size X (1 size = 100 shares)
    double meanBandSD = atof(argv[9]); // Standard deviation used for the "mean band"
    double tradeBandSD = atof(argv[10]); // Standard deviation used for the "trade band"
    double stopLossBandSD = atof(argv[11]); // Standard deviation used for the "stop loss band"
    bool verboseMode = (atoi(argv[12]) != 0); // Verbose mode

    std::cout.imbue(std::locale("en_US.UTF-8")); // For monetary amounts (std::put_money)
    std::cout << std::showbase;
    std::cout << std::fixed << std::setprecision(2); // Avoid scientific notation in large double values

    std::string clientID = "agent" + std::string(3 - std::to_string(clientNumber).length(), '0') + std::to_string(clientNumber);

    int64_t simulationDurationUS = simulationDurationS * 1'000'000.0; // Seconds to microseconds
    int64_t samplingFrequencyUS = samplingFrequencyS * 1'000'000.0; // Seconds to microseconds

    double positionRho = static_cast<double>(positionSizeY) / static_cast<double>(positionSizeX);
    positionSizeY = std::abs(positionSizeY);
    positionSizeX = std::abs(positionSizeX);

    double lastPriceY = 0.0;
    double lastPriceX = 0.0;

    std::deque<double> epsilons;
    double currentEpsilon = 0.0;
    double previousEpsilon = 0.0;

    double mean = 0.0;
    double sd = 0.0;
    double upperStopLossBand = 0.0;
    double upperTradeBand = 0.0;
    double upperMeanBand = 0.0;
    double lowerMeanBand = 0.0;
    double lowerTradeBand = 0.0;
    double lowerStopLossBand = 0.0;

    int currentPosition = 0;

    // Set up connection
    auto& initiator = shift::FIXInitiator::getInstance();

    RhoTrader client { clientID, { stockTickerY, stockTickerX } };
    // std::string outputFile = "output/" + clientID + "_" + RhoTrader::getLocalTime(true) + ".csv";
    // RhoTrader client { clientID, outputFile, { stockTickerY, stockTickerX } };

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
    // client.subOrderBook(stockTickerY);
    // client.subOrderBook(stockTickerX);

    std::chrono::microseconds elapsed;
    auto now = std::chrono::system_clock::now();
    auto begin = now;

    int64_t t = samplingFrequencyUS;

    // Waiting
    for (int i = 1; (lastPriceY == 0.0 || lastPriceX == 0.0) && t <= simulationDurationUS; ++i, t += samplingFrequencyUS) {
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1us - elapsed);

        lastPriceY = client.getLastPrice(stockTickerY);
        lastPriceX = client.getLastPrice(stockTickerX);

        if (verboseMode) {
            std::cout << "Waiting:" << std::setw(8) << i;
            std::cout << std::setprecision(6) << std::setw(15) << elapsed.count() / 1'000'000.0 << 's' << std::setprecision(2);
            std::cout << std::setw(12) << std::put_money(lastPriceY * 100.0) << std::setw(12) << std::put_money(lastPriceX * 100.0);
            std::cout << std::endl;
        }
    }

    // Training
    for (int i = 1; i <= samplingWindow && t <= simulationDurationUS; ++i, t += samplingFrequencyUS) {
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1us - elapsed);

        lastPriceY = client.getLastPrice(stockTickerY);
        lastPriceX = client.getLastPrice(stockTickerX);
        currentEpsilon = lastPriceY + positionRho * lastPriceX;
        epsilons.push_back(currentEpsilon);

        if (verboseMode) {
            std::cout << "Training:" << std::setw(7) << i;
            std::cout << std::setprecision(6) << std::setw(15) << elapsed.count() / 1'000'000.0 << 's' << std::setprecision(2);
            std::cout << std::setw(12) << std::put_money(lastPriceY * 100.0) << std::setw(12) << std::put_money(lastPriceX * 100.0);
            std::cout << std::setw(12) << currentEpsilon << std::endl;
        }
    }

    // Trading strategy
    while (t <= simulationDurationUS) {
        // Determine bands
        mean = simpleAverage(epsilons);
        sd = standardDeviation(epsilons, mean);

        upperStopLossBand = mean + sd * stopLossBandSD;
        upperTradeBand = mean + sd * tradeBandSD;
        upperMeanBand = mean + sd * meanBandSD;
        lowerMeanBand = mean - sd * meanBandSD;
        lowerTradeBand = mean - sd * tradeBandSD;
        lowerStopLossBand = mean - sd * stopLossBandSD;

        if (verboseMode) {
            std::cout << std::endl;
            std::cout << "Stats:" << std::endl;
            std::cout << "Mean: " << mean << std::endl;
            std::cout << "SD: " << sd << std::endl;
            std::cout << "Bands:" << std::endl;
            std::cout << "Upper Stop Loss: " << upperStopLossBand << std::endl;
            std::cout << "Upper Trade: " << upperTradeBand << std::endl;
            std::cout << "Upper Mean: " << upperMeanBand << std::endl;
            std::cout << "Lower Mean: " << lowerMeanBand << std::endl;
            std::cout << "Lower Trade: " << lowerTradeBand << std::endl;
            std::cout << "Lower Stop Loss: " << lowerStopLossBand << std::endl;
            std::cout << std::endl;
        }

        // Trading period
        now = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
        std::this_thread::sleep_for(t * 1us - elapsed);

        epsilons.pop_front();
        lastPriceY = client.getLastPrice(stockTickerY);
        lastPriceX = client.getLastPrice(stockTickerX);
        currentEpsilon = lastPriceY + positionRho * lastPriceX;
        epsilons.push_back(currentEpsilon);

        currentPosition = 0;

        if (verboseMode) {
            std::cout << "Trading:" << std::setw(8) << 1;
            std::cout << std::setprecision(6) << std::setw(15) << elapsed.count() / 1'000'000.0 << 's' << std::setprecision(2);
            std::cout << std::setw(12) << std::put_money(lastPriceY * 100.0) << std::setw(12) << std::put_money(lastPriceX * 100.0);
            std::cout << std::setw(12) << currentEpsilon << std::setw(4) << currentPosition << std::endl;
        }

        t += samplingFrequencyUS;
        previousEpsilon = currentEpsilon;

        for (int i = 2; i <= samplingWindow && t <= simulationDurationUS; ++i, t += samplingFrequencyUS, previousEpsilon = currentEpsilon) {
            now = std::chrono::system_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
            std::this_thread::sleep_for(t * 1us - elapsed);

            epsilons.pop_front();
            lastPriceY = client.getLastPrice(stockTickerY);
            lastPriceX = client.getLastPrice(stockTickerX);
            currentEpsilon = lastPriceY + positionRho * lastPriceX;
            epsilons.push_back(currentEpsilon);

            // Cross upper stop loss band from below
            if ((currentPosition == 1) && (currentEpsilon > upperStopLossBand) && (previousEpsilon < upperStopLossBand)) {
                currentPosition = 2;
                client.submitOrder({ shift::Order::MARKET_BUY, stockTickerY, positionSizeY });
                if (positionRho > 0) {
                    client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                } else {
                    client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                }
            }

            // Cross upper trade band from below
            else if ((currentPosition == 0) && (currentEpsilon > upperTradeBand) && (previousEpsilon < upperTradeBand)) {
                currentPosition = 1;
                client.submitOrder({ shift::Order::MARKET_SELL, stockTickerY, positionSizeY });
                if (positionRho > 0) {
                    client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                } else {
                    client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                }
            }

            // Cross upper mean band from above
            else if ((currentEpsilon < upperMeanBand) && (previousEpsilon > upperMeanBand)) {
                if (currentPosition == 1) {
                    client.submitOrder({ shift::Order::MARKET_BUY, stockTickerY, positionSizeY });
                    if (positionRho > 0) {
                        client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                    } else {
                        client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                    }
                }
                currentPosition = 0;
            }

            // Cross lower mean band from below
            else if ((currentEpsilon > lowerMeanBand) && (previousEpsilon < lowerMeanBand)) {
                if (currentPosition == -1) {
                    client.submitOrder({ shift::Order::MARKET_SELL, stockTickerY, positionSizeY });
                    if (positionRho > 0) {
                        client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                    } else {
                        client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                    }
                }
                currentPosition = 0;
            }

            // Cross lower trade band from above
            else if ((currentPosition == 0) && (currentEpsilon < lowerTradeBand) && (previousEpsilon > lowerTradeBand)) {
                currentPosition = -1;
                client.submitOrder({ shift::Order::MARKET_BUY, stockTickerY, positionSizeY });
                if (positionRho > 0) {
                    client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                } else {
                    client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                }
            }

            // Cross lower stop loss band from above
            else if ((currentPosition == -1) && (currentEpsilon < upperStopLossBand) && (previousEpsilon > upperStopLossBand)) {
                currentPosition = -2;
                client.submitOrder({ shift::Order::MARKET_SELL, stockTickerY, positionSizeY });
                if (positionRho > 0) {
                    client.submitOrder({ shift::Order::MARKET_SELL, stockTickerX, positionSizeX });
                } else {
                    client.submitOrder({ shift::Order::MARKET_BUY, stockTickerX, positionSizeX });
                }
            }

            if (verboseMode) {
                std::cout << "Trading:" << std::setw(8) << i;
                std::cout << std::setprecision(6) << std::setw(15) << elapsed.count() / 1'000'000.0 << 's' << std::setprecision(2);
                std::cout << std::setw(12) << std::put_money(lastPriceY * 100.0) << std::setw(12) << std::put_money(lastPriceX * 100.0);
                std::cout << std::setw(12) << currentEpsilon << std::setw(4) << currentPosition << std::endl;
            }
        }

        // Clear any remaining positions after trading period
        for (const auto& [ticker, item] : client.getPortfolioItems()) {
            if (item.getShares() > 0) {
                client.submitOrder({ shift::Order::MARKET_SELL, ticker, item.getShares() / (100) });
            } else if (item.getShares() < 0) {
                client.submitOrder({ shift::Order::MARKET_BUY, ticker, item.getShares() / (-100) });
            }
        }
    }

    std::this_thread::sleep_for(samplingFrequencyUS * 1us);

    initiator.disconnectBrokerageCenter();

    return 0;
}
