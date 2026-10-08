#include "neuroswap/strategy/strategy.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
namespace ns = neuroswap::strategy;

volatile std::sig_atomic_t g_stop_signal = 0;
void on_stop_signal(int signal_number) { g_stop_signal = signal_number; }

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " --model PATH --strategy baseline|prefetch|pin --report PATH\n"
        << "       [--fraction F] [--wait-resident P [--wait-timeout S] [--wait-plateau-s S]]\n"
        << "       [--cgroup DIR] [--hold]\n\n"
        << "Applies one memory strategy to a GGUF model file from a separate process.\n\n"
        << "  --model PATH      GGUF model file (mapped read-only, MAP_SHARED)\n"
        << "  --strategy S      baseline: no advice, no lock; residency observed only\n"
        << "                    prefetch: madvise(MADV_WILLNEED) over the whole file (advisory)\n"
        << "                    pin: touch + mlock() the first F of the per-token weight set\n"
        << "                         (all GGUF tensors except token_embd.weight, file-offset order)\n"
        << "  --fraction F      pin only, 0 < F <= 1\n"
        << "  --wait-resident P prefetch only: after madvise(), poll mincore() every 10 ms until\n"
        << "                    at least P percent (0 < P <= 100) of the file is resident\n"
        << "  --wait-timeout S  prefetch only: stop waiting after S seconds (default 120)\n"
        << "  --wait-plateau-s S  prefetch only: stop waiting once resident pages have not grown\n"
        << "                    for S seconds (default 3), e.g. when the file does not fit under\n"
        << "                    memory.max. residency_wait.stopped_reason records target, plateau\n"
        << "                    or timeout; none of them is an error\n"
        << "  --cgroup DIR      cgroup v2 directory to join before touching any page; its\n"
        << "                    memory.max sets room(L) (needs permission to write cgroup.procs)\n"
        << "  --hold            after setup print READY and keep pages mapped/locked until stdin\n"
        << "                    closes or SIGTERM/SIGINT; the report is then rewritten with a\n"
        << "                    final mincore() snapshot and the munlock() results\n"
        << "  --report PATH     JSON report (source = \"neuroswap_measured\")\n\n"
        << "pin_cap = min(room(L), 740.8 MiB RLIMIT_MEMLOCK policy, this process's RLIMIT_MEMLOCK),\n"
        << "room(L) = memory.max - peak_anon - other_rss - margin (docs/STRATEGY_DESIGN.md).\n"
        << "A pin above pin_cap is refused before any mlock(), with no fallback to baseline.\n\n"
        << "Exit codes: 0 ok, 1 error, 2 usage error, 3 pin refused by pin_cap.\n";
}

double parse_number(const std::string& value, const char* option) {
    std::size_t used = 0;
    double number = 0.0;
    try {
        number = std::stod(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size() || !std::isfinite(number)) {
        throw std::invalid_argument(std::string(option) + " must be a number");
    }
    return number;
}

double parse_fraction(const std::string& value) {
    std::size_t used = 0;
    double fraction = 0.0;
    try {
        fraction = std::stod(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size() || !std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0) {
        throw std::invalid_argument("--fraction must be a number with 0 < F <= 1");
    }
    return fraction;
}

bool write_report(const std::string& path, const ns::StrategyReport& report) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::trunc);
        if (!(file << ns::to_json(report))) return false;
    }
    return std::rename(temporary.c_str(), path.c_str()) == 0;
}

// Blocks until stdin reaches EOF or SIGTERM/SIGINT arrives. The signals stay
// blocked except inside ppoll(), so none can slip in between check and wait.
std::string wait_for_release() {
    sigset_t blocked{};
    sigset_t original{};
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGTERM);
    sigaddset(&blocked, SIGINT);
    sigprocmask(SIG_BLOCK, &blocked, &original);
    struct sigaction action {};
    action.sa_handler = on_stop_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);

    char buffer[256];
    while (g_stop_signal == 0) {
        pollfd input{STDIN_FILENO, POLLIN, 0};
        if (ppoll(&input, 1, nullptr, &original) < 0) {
            if (errno == EINTR) continue;
            return "poll_error";
        }
        if ((input.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            const ssize_t count = read(STDIN_FILENO, buffer, sizeof buffer);
            if (count == 0) return "stdin_closed";
            if (count < 0 && errno != EINTR) return "stdin_error";
        }
    }
    return g_stop_signal == SIGTERM ? "SIGTERM" : "SIGINT";
}
}  // namespace

int main(int argc, char* argv[]) {
    ns::StrategyConfig config;
    std::string report_path;
    bool hold = false;
    bool has_strategy = false;
    bool has_wait_timeout = false;
    bool has_wait_plateau = false;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help") { usage(argv[0]); return ns::kExitOk; }
            if (argument == "--hold") { hold = true; continue; }
            if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
            const std::string value = argv[++index];
            if (argument == "--model") config.model_path = value;
            else if (argument == "--strategy") { config.strategy = ns::parse_strategy(value); has_strategy = true; }
            else if (argument == "--fraction") config.fraction = parse_fraction(value);
            else if (argument == "--cgroup") config.cgroup_path = value;
            else if (argument == "--wait-resident") {
                const double percent = parse_number(value, "--wait-resident");
                if (percent <= 0.0 || percent > 100.0) throw std::invalid_argument("--wait-resident must be 0 < P <= 100");
                config.wait_resident_percent = percent;
            } else if (argument == "--wait-timeout") {
                config.wait_timeout_s = parse_number(value, "--wait-timeout");
                if (config.wait_timeout_s <= 0.0) throw std::invalid_argument("--wait-timeout must be > 0");
                has_wait_timeout = true;
            } else if (argument == "--wait-plateau-s") {
                config.wait_plateau_s = parse_number(value, "--wait-plateau-s");
                if (config.wait_plateau_s <= 0.0) throw std::invalid_argument("--wait-plateau-s must be > 0");
                has_wait_plateau = true;
            }
            else if (argument == "--report") report_path = value;
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (config.model_path.empty() || report_path.empty() || !has_strategy) {
            throw std::invalid_argument("--model, --strategy and --report are required");
        }
        if (config.strategy == ns::Strategy::Pin && !config.fraction) {
            throw std::invalid_argument("--strategy pin requires --fraction");
        }
        if (config.strategy != ns::Strategy::Pin && config.fraction) {
            throw std::invalid_argument("--fraction is only valid with --strategy pin");
        }
        const bool any_wait = config.wait_resident_percent || has_wait_timeout || has_wait_plateau;
        if (config.strategy != ns::Strategy::Prefetch && any_wait) {
            throw std::invalid_argument("--wait-resident, --wait-timeout and --wait-plateau-s are only valid "
                                        "with --strategy prefetch");
        }
        if ((has_wait_timeout || has_wait_plateau) && !config.wait_resident_percent) {
            throw std::invalid_argument("--wait-timeout and --wait-plateau-s require --wait-resident");
        }
    } catch (const std::exception& error) {
        std::cerr << "neuroswap_strategy: " << error.what() << "\n\n";
        usage(argv[0]);
        return ns::kExitUsage;
    }

    ns::StrategySession session(config);
    const int exit_code = session.run();
    if (!write_report(report_path, session.report())) {
        std::cerr << "neuroswap_strategy: cannot write report " << report_path << '\n';
        return ns::kExitError;
    }
    if (exit_code != ns::kExitOk) {
        std::cerr << "neuroswap_strategy: " << session.report().error << '\n';
        return exit_code;
    }
    if (!hold) return ns::kExitOk;

    std::cout << "READY " << report_path << std::endl;
    const auto held_from = std::chrono::steady_clock::now();
    const auto released_by = wait_for_release();
    const std::chrono::duration<double, std::milli> held = std::chrono::steady_clock::now() - held_from;
    session.release(released_by, held.count());
    if (!write_report(report_path, session.report())) {
        std::cerr << "neuroswap_strategy: cannot write final report " << report_path << '\n';
        return ns::kExitError;
    }
    return ns::kExitOk;
}
