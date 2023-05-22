#pragma once

#include <fstream>

#include <shift/coreclient/CoreClient.h>

class ZITrader : public shift::CoreClient {
public:
    static auto getLocalTime(bool underscore = false) -> std::string;

    ZITrader(std::string username,
        std::initializer_list<std::string> symbols = {});
    ZITrader(std::string username,
        std::string wealthTrackingFileName,
        std::initializer_list<std::string> symbols = {});
    ~ZITrader() = default;

protected:
    void receiveLastPrice(const std::string& symbol) override;
    void receiveExecution(const std::string& orderID) override;
    void receivePortfolioSummary() override;
    void receivePortfolioItem(const std::string& symbol) override;
    void receiveWaitingList() override;

private:
    std::ofstream m_wealthTrackingFile;
    std::vector<std::string> m_symbols;
};
