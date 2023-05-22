#include "MACDTrader.h"

#include <shift/miscutils/crossguid/Guid.h>

// Constructor ----------------------------------------------------------------

MACDTrader::MACDTrader(std::string username)
    : CoreClient { std::move(username) }
    , m_initialBuyingPower { 0.0 }
{
}

// Initial Portfolio ----------------------------------------------------------

auto MACDTrader::getInitialBuyingPower() -> double
{
    return m_initialBuyingPower;
}

auto MACDTrader::getInitialPrice(const std::string& symbol) -> double
{
    return m_initialPortfolio[symbol].getPrice();
}

auto MACDTrader::getInitialShares(const std::string& symbol) -> int
{
    return m_initialPortfolio[symbol].getShares();
}

auto MACDTrader::getDiffShares(const std::string& symbol) -> int
{
    return getPortfolioItem(symbol).getShares() - m_initialPortfolio[symbol].getShares();
}

// Mid Prices -----------------------------------------------------------------

// NEED TO ADD CHECK WHEN PRICE == 0.0 (SANITY CHECK)
auto MACDTrader::getBestBid(const std::string& symbol) -> std::pair<double, int>
{
    std::pair<double, int> bestBid;

    bestBid.first = getBestPrice(symbol).getBidPrice();
    bestBid.second = getBestPrice(symbol).getBidSize();

    if (isVerbose()) {
        std::cout << std::endl
                  << "Best Bid: " << symbol << " " << bestBid.first << " " << bestBid.second << std::endl;
    }

    return bestBid;
}

// NEED TO ADD CHECK WHEN PRICE == 0.0 (SANITY CHECK)
auto MACDTrader::getBestAsk(const std::string& symbol) -> std::pair<double, int>
{
    std::pair<double, int> bestAsk;

    bestAsk.first = getBestPrice(symbol).getAskPrice();
    bestAsk.second = getBestPrice(symbol).getAskSize();

    if (isVerbose()) {
        std::cout << std::endl
                  << "Best Ask: " << symbol << " " << bestAsk.first << " " << bestAsk.second << std::endl;
    }

    return bestAsk;
}

auto MACDTrader::getMidPrice(const std::string& symbol) -> double
{
    double bestBidPrice = getBestBid(symbol).first;
    double bestAskPrice = getBestAsk(symbol).first;

    if (bestBidPrice != 0.0 && bestAskPrice != 0.0) // sanity check
    {
        m_midPrice[symbol] = (bestBidPrice + bestAskPrice) / 2.0;
    }

    // if (isVerbose()) {
    //     std::cout << std::endl
    //               << "Mid Price: " << symbol << " " << m_midPrice[symbol] << std::endl;
    // }

    return m_midPrice[symbol];
}

// Submit Order (just to add verbose mode) ------------------------------------

void MACDTrader::createAndSubmitOrder(shift::Order::Type type, const std::string& symbol, int size, double price /* = 0.0 */)
{
    shift::Order order { type, symbol, size, price };

    if (isVerbose()) {
        switch (type) {
        case shift::Order::LIMIT_BUY:
            std::cout << std::endl
                      << "Limit Buy: " << order.getSymbol() << " " << order.getSize() << " " << order.getPrice() << std::endl;
            break;
        case shift::Order::LIMIT_SELL:
            std::cout << std::endl
                      << "Limit Sell: " << order.getSymbol() << " " << order.getSize() << " " << order.getPrice() << std::endl;
            break;
        case shift::Order::MARKET_BUY:
            std::cout << std::endl
                      << "Market Buy: " << order.getSymbol() << " " << order.getSize() << std::endl;
            break;
        case shift::Order::MARKET_SELL:
            std::cout << std::endl
                      << "Market Sell: " << order.getSymbol() << " " << order.getSize() << std::endl;
            break;
        default:
            break;
        }
    }

    submitOrder(order);
}

// Virtual member functions from shift::CoreClient ----------------------------

void MACDTrader::receiveLastPrice(const std::string& symbol)
{
    // if (isVerbose()) {
    //     std::cout << symbol << " Trade: " << getLastPrice(symbol) << std::endl;
    // }
}

void MACDTrader::receiveExecution(const std::string& orderID)
{
    // if (isVerbose()) {
    //     auto order = getOrder(orderID);
    //     double price = 0.0;
    //     if (order.getExecutedSize() == order.getSize()) {
    //         price = order.getExecutedPrice();
    //     } else {
    //         price = order.getPrice();
    //     }
    //     std::cout << "Report:" << std::endl;
    //     std::cout << order.getSymbol() << " " << order.getTypeString() << " " << price << " " << order.getSize() << " "
    //               << order.getExecutedSize() << " " << order.getID() << " " << order.getStatusString() << " " << std::endl;
    // }
}

void MACDTrader::receivePortfolioSummary()
{
    if (m_initialBuyingPower <= 0.0) {
        m_initialBuyingPower = getPortfolioSummary().getTotalBP();

        if (isVerbose()) {
            std::cout << "Buying Power: " << m_initialBuyingPower << std::endl;
        }
    }
}

void MACDTrader::receivePortfolioItem(const std::string& symbol)
{
    if (m_initialBuyingPower == getPortfolioSummary().getTotalBP()) { // first check (we need a better one)
        if (m_initialPortfolio.count(symbol) == 0) // second check (bad fix)
        {
            m_initialPortfolio[symbol] = getPortfolioItem(symbol);
        }
    }

    if (isVerbose()) {
        std::cout << symbol << " Shares: " << getPortfolioItem(symbol).getShares() << std::endl;
    }
}

void MACDTrader::receiveWaitingList()
{
    // if (isVerbose()) {
    //     if (getWaitingListSize() > 0) {
    //         auto waitingList = getWaitingList();
    //         std::cout << "Waiting List:" << std::endl;
    //         for (auto order : waitingList) {
    //             std::cout << order.getSymbol() << " " << order.getTypeString() << " " << order.getPrice() << " " << order.getSize() << " "
    //                       << order.getExecutedSize() << " " << order.getID() << " " << order.getStatusString() << " " << std::endl;
    //         }
    //     } else {
    //         std::cout << "Waiting List Empty!" << std::endl;
    //     }
    // }
}

// ----------------------------------------------------------------------------
