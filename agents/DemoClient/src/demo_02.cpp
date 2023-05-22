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

    Order aaplLimitBuy { Order::LIMIT_BUY, "AAPL", 10, 10.00 };
    client.submitOrder(aaplLimitBuy);

    Order msftLimitBuy { Order::LIMIT_BUY, "MSFT", 10, 10.00 };
    client.submitOrder(msftLimitBuy);

    // ------------------------------------------------------------------------

    cout << "Press enter to stop the demo." << endl;
    getchar();

    initiator.disconnectBrokerageCenter();

    return 0;
}
