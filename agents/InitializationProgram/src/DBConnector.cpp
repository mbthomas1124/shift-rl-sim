#include "DBConnector.h"

#include <sstream>

#include <shift/miscutils/crypto/Decryptor.h>
#include <shift/miscutils/database/Common.h>
#include <shift/miscutils/terminal/Common.h>

DBConnector::DBConnector()
    : m_pConn(nullptr)
{
}

DBConnector::~DBConnector()
{
    disconnectDB();
}

/*static*/ auto DBConnector::getInstance() -> DBConnector&
{
    static DBConnector s_DBInst;
    return s_DBInst;
}

auto DBConnector::init(const std::string& cryptoKey, const std::string& fileName) -> bool
{
    m_loginInfo = shift::crypto::readEncryptedConfigFile(cryptoKey, fileName);
    return !m_loginInfo.empty();
}

/* Establish connection to database */
auto DBConnector::connectDB() -> bool
{
    disconnectDB();

    const auto connInfo = "hostaddr=" + m_loginInfo["DBHostaddr"] + " port=" + m_loginInfo["DBPort"] + " dbname=" + m_loginInfo["DBname"] + " user=" + m_loginInfo["DBUser"] + " password=" + m_loginInfo["DBPassword"];
    m_pConn = PQconnectdb(connInfo.c_str());
    if (PQstatus(m_pConn) != CONNECTION_OK) {
        disconnectDB();
        cout << COLOR_ERROR "ERROR: Connection to database failed.\n" NO_COLOR;
        return false;
    }

    return shift::database::checkCreateTable<shift::database::PortfolioSummary>(m_pConn)
        && shift::database::checkCreateTable<shift::database::PortfolioItem>(m_pConn);
}

void DBConnector::disconnectDB()
{
    if (m_pConn == nullptr) {
        return;
    }

    PQfinish(m_pConn);
    m_pConn = nullptr;
}

auto DBConnector::checkTableExist(const std::string& tableName) const -> bool
{
    return shift::database::checkTableExist(m_pConn, tableName) == shift::database::TABLE_STATUS::EXISTS;
}

auto DBConnector::doQuery(std::string query, std::string msgIfStatMismatch, ExecStatusType statToMatch /* = PGRES_COMMAND_OK */, PGresult** ppRes /* = nullptr */) -> bool
{
    return shift::database::doQuery(m_pConn, std::move(query), std::move(msgIfStatMismatch), statToMatch, ppRes);
}

void DBConnector::initializeOneRecord(const std::string& username, const double buyingPower, const std::string& symbol, const int shares, const double price)
{
    auto userID = shift::database::readFieldsOfRow(m_pConn, "SELECT id FROM traders WHERE username = '" + username + "';", 1);
    if (userID.empty()) {
        cout << COLOR_WARNING "WARNING: There is no user called '" << username << "' in traders table, skipped." NO_COLOR << endl;
        return;
    }

    const auto bpStr = std::to_string(buyingPower);
    const auto shStr = std::to_string(shares);
    const auto prStr = std::to_string(price);

    auto summaryOfUser = shift::database::readFieldsOfRow(m_pConn, "SELECT portfolio_summary.id, portfolio_summary.buying_power, portfolio_summary.total_shares FROM portfolio_summary INNER JOIN traders ON portfolio_summary.id = traders.id WHERE username = '" + username + "';", 3);
    if (summaryOfUser.empty()) { // user absent in portfolio_summary ?
        if (!doQuery("INSERT INTO portfolio_summary (id, buying_power, total_shares) VALUES ('" + userID.front() + "'," + bpStr + ',' + shStr + ");", COLOR_ERROR "Insert portfolio_summary error:" NO_COLOR)) {
            cout << PQerrorMessage(m_pConn) << endl;
            return;
        }
    } else {
        const auto newBPStr = std::to_string(std::stod(summaryOfUser[1]) + buyingPower);
        const auto newShStr = std::to_string(std::stoi(summaryOfUser[2]) + shares);
        if (!doQuery("UPDATE portfolio_summary SET buying_power = " + newBPStr + ", total_shares = " + newShStr + " WHERE id = '" + summaryOfUser[0] + "';", COLOR_ERROR "Update portfolio_summary error:" NO_COLOR)) {
            cout << PQerrorMessage(m_pConn) << endl;
            return;
        }
    }

    cout << COLOR_WARNING << username << " has touched the portfolio_summary." NO_COLOR << endl;

    auto itemOfUser = shift::database::readFieldsOfRow(m_pConn, "SELECT portfolio_items.id, portfolio_items.symbol, portfolio_items.long_price, portfolio_items.long_shares FROM portfolio_items INNER JOIN traders ON portfolio_items.id = traders.id WHERE username = '" + username + "' AND symbol = '" + symbol + "';", 4);
    if (itemOfUser.empty()) { // user-symbol combination absent in portfolio_items ?
        if (!doQuery("INSERT INTO portfolio_items (id, symbol, long_price, long_shares) VALUES ((SELECT id FROM traders WHERE username = '" + username + "'),'" + symbol + "'," + prStr + ',' + shStr + ");", COLOR_ERROR "Insert portfolio_items error:" NO_COLOR)) {
            cout << PQerrorMessage(m_pConn) << endl;
            return;
        }
    } else {
        const auto oldLP = std::stod(itemOfUser[2]);
        const auto oldLS = std::stoi(itemOfUser[3]);

        const auto newLPRaw = (oldLP * oldLS + price * shares) / (oldLS + shares);
        // cout << '(' << oldLP << " * " << oldLS << " + " << price << " * " << shares << ") / (" << oldLS << " + " << shares << ") == " << newLPRaw << endl;
        std::stringstream ss;
        ss << std::fixed;
        ss.precision(2); // set # places after decimal
        ss << newLPRaw;

        const auto newLPStr = ss.str();
        const auto newLSStr = std::to_string(oldLS + shares);

        if (!doQuery("UPDATE portfolio_items SET long_price = " + newLPStr + ", long_shares = " + newLSStr + " WHERE id = '" + itemOfUser[0] + "' AND symbol = '" + itemOfUser[1] + "';", COLOR_ERROR "Update portfolio_items error:" NO_COLOR)) {
            cout << PQerrorMessage(m_pConn) << endl;
            return;
        }
    }

    cout << COLOR_WARNING << username << " has touched the portfolio_items.\n" NO_COLOR << endl;
}
