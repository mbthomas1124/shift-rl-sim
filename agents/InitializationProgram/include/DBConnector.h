#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

#include <libpq-fe.h>

class DBConnector {
public:
    virtual ~DBConnector();

    static auto getInstance() -> DBConnector&;

    auto init(const std::string& cryptoKey, const std::string& fileName) -> bool;

    auto connectDB() -> bool;
    void disconnectDB();

    auto checkTableExist(const std::string& tableName) const -> bool;
    auto doQuery(std::string query, std::string msgIfStatMismatch, ExecStatusType statToMatch = PGRES_COMMAND_OK, PGresult** ppRes = nullptr) -> bool;
    void initializeOneRecord(const std::string& username, const double buyingPower, const std::string& symbol, const int shares, const double price);

protected:
    PGconn* m_pConn;
    mutable std::mutex m_mtxPSQL;

private:
    DBConnector(); // singleton pattern
    DBConnector(const DBConnector&) = delete; // forbid copying
    auto operator=(const DBConnector&) -> DBConnector& = delete; // forbid assigning

    std::unordered_map<std::string, std::string> m_loginInfo;
};
