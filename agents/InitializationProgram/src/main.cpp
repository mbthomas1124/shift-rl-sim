#include "DBConnector.h"

#include <algorithm>
#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <fstream>
#include <iostream>
#include <random>

#include <pwd.h>

#include <boost/program_options.hpp>

#include <shift/miscutils/terminal/Common.h>

#define CSTR_HELP \
    "help"
#define CSTR_CONFIG \
    "config"
#define CSTR_KEY \
    "key"
#define CSTR_DBLOGIN_TXT \
    "dbLogin.txt"
#define CSTR_RESET \
    "reset"
#define CSTR_TICKER \
    "ticker"
#define CSTR_BEGIN \
    "begin"
#define CSTR_END \
    "end"
#define CSTR_SIZE \
    "size"
#define CSTR_PRICE \
    "price"
#define CSTR_ALPHA \
    "alpha"
#define CSTR_SEED \
    "seed"
#define CSTR_REPEAT \
    "repeat"

namespace po = boost::program_options;

struct trader {
    std::string username;
    double probability;
    double buyingPower;
    int size;
};

auto createRNG(const std::string& filename, bool load = false) -> std::mt19937
{
    std::mt19937 rng { std::random_device {}() };

    if (load) {
        std::ifstream inputFile { filename, std::ios::binary };
        if (inputFile) {
            inputFile >> rng;
        } else {
            load = false;
        }
    }

    if (!load) {
        std::ofstream outputFile { filename, std::ios::binary };
        outputFile << rng;
    }

    return rng;
}

auto main(int argc, char** argv) -> int
{
    struct {
        std::string configDir;
        std::string cryptoKey;
        int traderNumBeg;
        int traderNumEnd;
        std::string initTicker;
        double initPrice;
        int nTotalSize;
        std::string randomSeedPath;
        bool repeatRandomSeed;
    } params = {
        "/usr/local/share/shift-research/InitializationProgram/", // default folder for configuration files
        "SHIFT123", // built-in initial crypto key used for encrypting dbLogin.txt
        0,
        0,
        "CS1",
        100.0,
        0,
        "rng",
        false,
    };

    po::options_description desc("\nUSAGE: ./InitializationProgram [options] <args>\n\n\tThis is the InitializationProgram.\n\nOPTIONS");
    desc.add_options() // <--- every line-end from here needs a comment mark so that to prevent auto formating into single line
        (CSTR_HELP ",h", "show help") //
        (CSTR_CONFIG ",c", po::value<std::string>(), "set configuration directory containing the " CSTR_DBLOGIN_TXT) //
        (CSTR_KEY ",k", po::value<std::string>(), "key of " CSTR_DBLOGIN_TXT " file to decrpyt") //
        (CSTR_RESET ",r", "reset client portfolio records") //
        (CSTR_BEGIN ",b", po::value<int>(), "beginning number of all trader names' number-suffix, e.g. '2' in 'agent002, ..., agent090';\ndefault is 0") //
        (CSTR_END ",e", po::value<int>(), "ending number of all trader names' number-suffix, inclusively, e.g. '90' in 'agent002, ..., agent090';\ndefault is 0;\nshall not be less than the beginning number (--" CSTR_BEGIN ");\nif it's equal to the beginning number, only a single trader will be processed") //
        (CSTR_TICKER ",t", po::value<std::string>(), "ticker name") //
        (CSTR_PRICE ",p", po::value<double>(), "ticker price") //
        (CSTR_SIZE ",s", po::value<int>(), "total available size (volume) for all traders (1 size == 100 shares)") //
        (CSTR_ALPHA ",a", po::value<double>(), "alpha parameter of the Dirichlet distribution") //
        (CSTR_SEED ",n", po::value<std::string>(), "random seed path (to store state)") //
        (CSTR_REPEAT ",y", "repeat random seed") //
        ; // add_options

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const boost::program_options::error& e) {
        cerr << COLOR_ERROR "ERROR: " << e.what() << NO_COLOR << endl;
        return 1;
    } catch (...) {
        cerr << COLOR_ERROR "ERROR: Exception of unknown type!" NO_COLOR << endl;
        return 2;
    }

    if (vm.count(CSTR_HELP) > 0) {
        cout << '\n'
             << desc << '\n'
             << endl;
        return 0;
    }

    if (vm.count(CSTR_CONFIG) > 0) {
        params.configDir = vm[CSTR_CONFIG].as<std::string>();
        if (params.configDir.length() > 0 && params.configDir.back() != '/') {
            params.configDir.push_back('/');
        }

        cout << COLOR "Configuration directory was set to: "
             << params.configDir << ".\n" NO_COLOR << endl;
    } else {
        cout << COLOR "Configuration directory was not set, default config-dir is used: " << params.configDir << NO_COLOR << '\n'
             << endl;
    }

    if (vm.count(CSTR_KEY) > 0) {
        params.cryptoKey = vm[CSTR_KEY].as<std::string>();
    } else {
        cout << COLOR "The built-in initial key 'SHIFT123' is used for reading encrypted login files." NO_COLOR << '\n'
             << endl;
    }

    // Database parameters
    DBConnector::getInstance().init(params.cryptoKey, params.configDir + CSTR_DBLOGIN_TXT);

    if (!DBConnector::getInstance().connectDB()) {
        cout << COLOR_ERROR "Connect DB error!" NO_COLOR << endl;
    }

    if (vm.count(CSTR_RESET) > 0) {
        cout << COLOR_WARNING "Resetting the databases..." NO_COLOR << endl;
        DBConnector::getInstance().doQuery("DROP TABLE portfolio_summary CASCADE", COLOR_ERROR "ERROR: Failed to drop [ portfolio_summary ]." NO_COLOR);
        DBConnector::getInstance().doQuery("DROP TABLE portfolio_items CASCADE", COLOR_ERROR "ERROR: Failed to drop [ portfolio_items ]." NO_COLOR);

        if (!DBConnector::getInstance().connectDB()) {
            cout << COLOR_ERROR "Connect DB error!" NO_COLOR << endl;
            return 3;
        }

        return 0;
    }

    if (vm.count(CSTR_BEGIN) > 0 && vm.count(CSTR_END) == 0) {
        params.traderNumBeg = params.traderNumEnd = vm[CSTR_BEGIN].as<int>();
    } else if (vm.count(CSTR_BEGIN) == 0 && vm.count(CSTR_END) > 0) {
        params.traderNumBeg = params.traderNumEnd = vm[CSTR_END].as<int>();
    } else if (vm.count(CSTR_BEGIN) > 0 && vm.count(CSTR_END) > 0) {
        params.traderNumBeg = vm[CSTR_BEGIN].as<int>();
        params.traderNumEnd = vm[CSTR_END].as<int>();
    } else {
        cout << COLOR_ERROR "ERROR: Please provide at least begin or end number-suffix!" NO_COLOR << endl;
        return 4;
    }

    if (params.traderNumEnd < params.traderNumBeg) {
        cout << COLOR_ERROR "ERROR: the trader's ending number-suffix shall not be smaller than the beginning one!" NO_COLOR << endl;
        return 5;
    }

    if (params.traderNumBeg < 0 || params.traderNumEnd > 999) {
        cout << COLOR_ERROR "ERROR: the trader's number-suffix shall be between 0 to 999!" NO_COLOR << endl;
        return 6;
    }

    if (vm.count(CSTR_TICKER) > 0) {
        params.initTicker = vm[CSTR_TICKER].as<std::string>();
    }

    if (vm.count(CSTR_PRICE) > 0) {
        params.initPrice = vm[CSTR_PRICE].as<double>();
    }

    if (vm.count(CSTR_SIZE) > 0) {
        params.nTotalSize = vm[CSTR_SIZE].as<int>();
    }

    if (vm.count(CSTR_SEED) > 0) {
        params.randomSeedPath = vm[CSTR_SEED].as<std::string>();
    }

    if (vm.count(CSTR_REPEAT) > 0) {
        params.repeatRandomSeed = true;
    }

    const int nTraders = params.traderNumEnd - params.traderNumBeg + 1;

    // Create ~/.shift/InitializationProgram directory if it does not exist
    const char* homeDir = nullptr;
    if ((homeDir = getenv("HOME")) == nullptr) {
        homeDir = getpwuid(getuid())->pw_dir;
    }
    std::string servicePath { homeDir };
    servicePath += "/.shift/InitializationProgram";
#if __has_include(<filesystem>)
    std::filesystem::create_directories(servicePath);
#else
    std::experimental::filesystem::create_directories(servicePath);
#endif

    // Flat Dirichlet Distribution parameters
    std::mt19937 gen = createRNG(servicePath + "/" + params.randomSeedPath, params.repeatRandomSeed);
    std::gamma_distribution<double> gm { vm.count(CSTR_ALPHA) > 0 ? vm[CSTR_ALPHA].as<double>() : static_cast<double>(nTraders), 1.0 };

    // Flat Dirichlet Distribution auxiliary variables
    double dirichlet = 0.0;
    double sumDirichlet = 0.0;

    // Traders data
    std::vector<trader> traders;
    int sizeDistrib = 0;
    int sizeRemain = 0;

    // Connect to database
    if (!DBConnector::getInstance().connectDB()) {
        cout << "Error, could not connect to database! Press enter to continue." << endl;
    } else {
        cout << "Successfully connected to database! Press enter to continue." << endl;
    }
    getchar();

    // Flat Dirichlet Distribution
    constexpr int nMaxSuffixDigits = 3; // agent[000 - 999]
    for (int i = params.traderNumBeg; i <= params.traderNumEnd; ++i) {
        dirichlet = gm(gen);
        auto numStr = std::to_string(i);
        traders.push_back({ "agent" + std::string(nMaxSuffixDigits - numStr.length(), '0') + numStr, dirichlet, 0.0, 0 }); // e.g. agent002
        sumDirichlet += dirichlet;
    }
    for (auto& trader : traders) {
        trader.probability = trader.probability / sumDirichlet;
        trader.size = std::floor(trader.probability * params.nTotalSize);
        sizeDistrib += trader.size;
    }

    // Distribute remaining shares after initial division (in a really stupid way)
    std::sort(traders.begin(), traders.end(), [](auto const& a, auto const& b) {
        return a.probability < b.probability;
    });

    sizeRemain = params.nTotalSize - sizeDistrib;
    while (sizeRemain > 0) {
        sizeRemain = std::min(sizeRemain, nTraders);
        for (int i = 0; i < sizeRemain; ++i) {
            ++traders[i].size;
            ++sizeDistrib;
        }
        sizeRemain = params.nTotalSize - sizeDistrib;
    }

    // Store values in the database
    std::sort(traders.begin(), traders.end(), [](auto const& a, auto const& b) {
        return a.username < b.username;
    });

    for (auto& trader : traders) {
        trader.buyingPower = params.initPrice * trader.size * 100;
        DBConnector::getInstance().initializeOneRecord(trader.username, trader.buyingPower, params.initTicker, trader.size * 100, params.initPrice);
    }

    cout << "Done! Press enter to finish." << endl;
    std::getchar();

    return 0;
}
