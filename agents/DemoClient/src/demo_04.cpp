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

    cout << "Symbol" << '\t' // header
         << "Type" << '\t'
         << "Price" << '\t'
         << "Size" << '\t'
         << "ID" << endl;

    for (const auto& order : client.getWaitingList()) { // content
        cout << order.getSymbol() << '\t'
             << order.getType() << '\t'
             << order.getPrice() << '\t'
             << order.getSize() << '\t'
             << order.getID() << endl;
    }

    cout << endl; // pretty-printing

    cout << "Waiting List Length: " << client.getWaitingListSize() << endl;

    cout << "Canceling all pending orders... ";

    client.cancelAllPendingOrders();

    int i = 0;
    while (client.getWaitingListSize() > 0) {
        cout << ++i << "... ";
        this_thread::sleep_for(1s);
    }

    cout << endl;

    cout << "Waiting List Length: " << client.getWaitingListSize() << endl;

    cout << endl; // pretty-printing

    // ------------------------------------------------------------------------

    cout << "Press enter to stop the demo." << endl;
    getchar();

    initiator.disconnectBrokerageCenter();

    return 0;
}
