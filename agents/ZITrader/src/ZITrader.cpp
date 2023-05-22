#include "ZITrader.h"

#include <cmath>
#include <sstream>

/* static */ auto ZITrader::getLocalTime(bool underscore) -> std::string
{
    std::ostringstream oss;

    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now); // Loses microseconds information
    auto us = std::to_string(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now - std::chrono::system_clock::from_time_t(now_t))
            .count());

    if (underscore) {
        oss << std::put_time(std::localtime(&now_t), "%F_%T");
    } else {
        oss << std::put_time(std::localtime(&now_t), "%F %T");
    }
    oss << '.' << std::string(6 - us.length(), '0') + us; // Microseconds with "0" padding

    return oss.str();
}

ZITrader::ZITrader(std::string username,
    std::initializer_list<std::string> symbols /* = {} */)
    : CoreClient { std::move(username) }
    , m_wealthTrackingFile { "" }
    , m_symbols { std::move(symbols) }
{
}

ZITrader::ZITrader(std::string username,
    std::string wealthTrackingFileName,
    std::initializer_list<std::string> symbols /* = {} */)
    : CoreClient { std::move(username) }
    , m_wealthTrackingFile { std::move(wealthTrackingFileName) }
    , m_symbols { std::move(symbols) }
{
    if (m_wealthTrackingFile) {
        // Price should always have two decimal places
        m_wealthTrackingFile << std::fixed << std::setprecision(2);
        m_wealthTrackingFile << "real_time,asset,shares,price" << std::endl;
    }
}

void ZITrader::receiveLastPrice(const std::string& symbol)
{
    if (isVerbose()) {
        if (std::find(m_symbols.begin(), m_symbols.end(), symbol) != m_symbols.end()) {
            std::cout << symbol << " Trade: "
                      << std::put_money(getLastPrice(symbol) * 100.0) << std::endl;
        }
    }
}

void ZITrader::receiveExecution(const std::string& orderID)
{
    if (isVerbose()) {
        auto order = getOrder(orderID);
        double price = 0.0;
        if (order.getStatus() == shift::Order::Status::FILLED) {
            price = order.getExecutedPrice();
        } else {
            price = order.getPrice();
        }
        std::cout << "Report:" << std::endl;
        std::cout << order.getSymbol() << ' ' << order.getTypeString() << ' '
                  << std::put_money(price * 100.0) << ' ' << order.getSize() << ' '
                  << order.getExecutedSize() << ' ' << order.getID() << ' '
                  << order.getStatusString() << ' ' << std::endl;
    }
}

void ZITrader::receivePortfolioSummary()
{
    if (isVerbose()) {
        std::cout << "Buying Power: "
                  << std::put_money(getPortfolioSummary().getTotalBP() * 100.0) << std::endl;
    }
    if (m_wealthTrackingFile) {
        m_wealthTrackingFile << getLocalTime() << ',';
        m_wealthTrackingFile << "CASH,1," << getPortfolioSummary().getTotalBP() << std::endl;
    }
}

void ZITrader::receivePortfolioItem(const std::string& symbol)
{
    if (isVerbose()) {
        std::cout << symbol << " Shares: " << getPortfolioItem(symbol).getShares() << std::endl;
    }
    if (m_wealthTrackingFile) {
        m_wealthTrackingFile << getLocalTime() << ',';
        m_wealthTrackingFile << symbol << ','
                             << getPortfolioItem(symbol).getShares() << ','
                             << getPortfolioItem(symbol).getPrice() << std::endl;
    }
}

void ZITrader::receiveWaitingList()
{
    if (isVerbose()) {
        if (getWaitingListSize() > 0) {
            auto waitingList = getWaitingList();
            std::cout << "Waiting List:" << std::endl;
            for (const auto& order : waitingList) {
                std::cout << order.getSymbol() << ' ' << order.getTypeString() << ' '
                          << std::put_money(order.getPrice() * 100.0) << ' ' << order.getSize() << ' '
                          << order.getExecutedSize() << ' ' << order.getID() << ' '
                          << order.getStatusString() << ' ' << std::endl;
            }
        } else {
            std::cout << "Waiting List Empty!" << std::endl;
        }
    }
}
