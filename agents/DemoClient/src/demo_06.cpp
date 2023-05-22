#include <thread>

#include <shift/coreclient/CoreClient.h>
#include <shift/coreclient/FIXInitiator.h>

using namespace std::chrono_literals;

using namespace shift;
using namespace std;

auto main(int argc, char** argv) -> int
{
    // Necessary "objects"
    auto& initiator = FIXInitiator::getInstance();
    CoreClient client { "democlient" };

    // Set up connection
    try {
        initiator.connectBrokerageCenter("initiator.cfg", &client, "password");
    } catch (const exception& e) {
        cout << "Something went wrong: " << e.what() << endl;
        cout << "Press enter to stop the demo." << endl;
        getchar();
        return 1;
    }

    // Subscribe to order book data
    client.subAllOrderBook();

    // Wait to receive all client information
    this_thread::sleep_for(1s);

    // ------------------------------------------------------------------------

    cout << endl; // pretty-printing

    cout << "Buying Power" << '\t' // header
         << "Total Shares" << '\t'
         << "Total P&L" << endl;

    cout << client.getPortfolioSummary().getTotalBP() << "\t\t" // content
         << client.getPortfolioSummary().getTotalShares() << "\t\t"
         << client.getPortfolioSummary().getTotalRealizedPL() << endl;

    cout << endl; // pretty-printing

    cout << "Symbol" << '\t' // header
         << "Shares" << '\t'
         << "Price" << '\t'
         << "P&L" << endl;

    for (const auto& [ticker, item] : client.getPortfolioItems()) { // content
        cout << ticker << '\t'
             << item.getShares() << '\t'
             << item.getPrice() << '\t'
             << item.getRealizedPL() << endl;
    }

    cout << endl; // pretty-printing

    // ------------------------------------------------------------------------

    cout << "Press enter to stop the demo." << endl;
    getchar();

    initiator.disconnectBrokerageCenter();

    return 0;
}
