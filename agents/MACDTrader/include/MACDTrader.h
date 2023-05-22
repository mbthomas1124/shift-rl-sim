#pragma once

#include <shift/coreclient/CoreClient.h>
#include <shift/coreclient/PortfolioItem.h>

class MACDTrader : public shift::CoreClient {
public:
    // Constructor and Destructor
    MACDTrader(std::string username);
    ~MACDTrader() = default;

    // Initial Portfolio
    auto getInitialBuyingPower() -> double;
    auto getInitialPrice(const std::string& symbol) -> double;
    auto getInitialShares(const std::string& symbol) -> int;
    auto getDiffShares(const std::string& symbol) -> int;

    // Bid, Ask, and Mid Prices
    auto getBestBid(const std::string& symbol) -> std::pair<double, int>;
    auto getBestAsk(const std::string& symbol) -> std::pair<double, int>;
    auto getMidPrice(const std::string& symbol) -> double;

    // Submit Order (just to add verbose mode)
    void createAndSubmitOrder(shift::Order::Type type, const std::string& symbol, int size, double price = 0.0);

protected:
    // Virtual member functions from shift::CoreClient

    void receiveLastPrice(const std::string& symbol) override;
    void receiveExecution(const std::string& orderID) override;
    void receivePortfolioSummary() override;
    void receivePortfolioItem(const std::string& symbol) override;
    void receiveWaitingList() override;

private:
    // Initial Portfolio
    double m_initialBuyingPower;
    std::map<std::string, shift::PortfolioItem> m_initialPortfolio;

    // Mid Prices
    std::map<std::string, double> m_midPrice;
};
