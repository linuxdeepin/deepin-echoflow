#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace crispasr_cli_process {

struct RunResult {
    std::string output;
    int exit_code = -1;
    bool timed_out = false;
    bool spawn_failed = false;
};

// Both quoters are compiled on every platform, and quote_arg() picks one. They
// used to sit in the two arms of an #ifdef, which meant the Windows rules were
// never compiled — let alone tested — anywhere else, and #328 is what that
// costs: sherpa was spawned with POSIX single quotes that cmd.exe does not
// interpret, so it received literal quote characters inside its model paths and
// wedged. Neither a Linux build nor a macOS build could have caught it, and the
// only machines that could were the ones already broken.
//
// Defined unconditionally, they are pure string functions that
// test-subprocess-quoting.cpp round-trips against reference parsers on any host.

// MSVC / CommandLineToArgvW rules: backslashes are literal unless they precede
// a quote, where they double and the quote is escaped. A run of backslashes at
// the very end of a quoted argument doubles too, so the closing quote is not
// swallowed by the last one.
inline std::string quote_arg_windows(const std::string& arg) {
    if (arg.empty())
        return "\"\"";
    if (arg.find_first_of(" \t\n\v\"") == std::string::npos)
        return arg;

    std::string out = "\"";
    size_t bs = 0;
    for (char c : arg) {
        if (c == '\\') {
            bs++;
        } else if (c == '"') {
            out.append(bs * 2 + 1, '\\');
            out.push_back('"');
            bs = 0;
        } else {
            out.append(bs, '\\');
            bs = 0;
            out.push_back(c);
        }
    }
    out.append(bs * 2, '\\');
    out.push_back('"');
    return out;
}

// POSIX sh: single quotes take everything literally, so only the single quote
// itself needs handling — close, emit an escaped one, reopen.
inline std::string quote_arg_posix(const std::string& arg) {
    std::string out = "'";
    for (char c : arg) {
        if (c == '\'')
            out += "'\\''";
        else
            out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

inline std::string quote_arg(const std::string& arg) {
#ifdef _WIN32
    return quote_arg_windows(arg);
#else
    return quote_arg_posix(arg);
#endif
}

inline std::string join_cmdline(const std::vector<std::string>& args) {
    std::string cmd;
    for (const auto& arg : args) {
        if (!cmd.empty())
            cmd.push_back(' ');
        cmd += quote_arg(arg);
    }
    return cmd;
}

inline int timeout_from_audio_samples(const char* env_name, int n_samples, int min_sec = 120, int per_audio_sec = 5) {
    const char* env = std::getenv(env_name);
    if (env && *env) {
        const int v = std::atoi(env);
        if (v > 0)
            return v;
    }
    const int audio_s = std::max(1, (int)std::ceil((double)n_samples / 16000.0));
    return std::max(min_sec, audio_s * per_audio_sec);
}

#ifdef _WIN32
inline RunResult run_capture_stdout(const std::vector<std::string>& args, int timeout_sec,
                                    bool capture_stderr = false) {
    RunResult result;
    if (args.empty()) {
        result.spawn_failed = true;
        return result;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE out_rd = NULL;
    HANDLE out_wr = NULL;
    if (!CreatePipe(&out_rd, &out_wr, &sa, 0)) {
        result.spawn_failed = true;
        return result;
    }
    SetHandleInformation(out_rd, HANDLE_FLAG_INHERIT, 0);

    HANDLE nul = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_READ, &sa, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, NULL);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out_wr;
    si.hStdError = capture_stderr ? out_wr : ((nul != INVALID_HANDLE_VALUE) ? nul : out_wr);

    PROCESS_INFORMATION pi{};
    std::string cmdline = join_cmdline(args);
    std::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(NULL, mutable_cmd.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(out_wr);
    if (nul != INVALID_HANDLE_VALUE)
        CloseHandle(nul);

    if (!ok) {
        CloseHandle(out_rd);
        result.spawn_failed = true;
        return result;
    }

    std::thread reader([&]() {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(out_rd, buf, sizeof(buf), &n, NULL) && n > 0)
            result.output.append(buf, buf + n);
    });

    const DWORD wait_ms = timeout_sec > 0 ? (DWORD)timeout_sec * 1000u : INFINITE;
    DWORD wait_rc = WaitForSingleObject(pi.hProcess, wait_ms);
    if (wait_rc == WAIT_TIMEOUT) {
        result.timed_out = true;
        TerminateProcess(pi.hProcess, 124);
        WaitForSingleObject(pi.hProcess, INFINITE);
    }

    DWORD exit_code = 0;
    if (GetExitCodeProcess(pi.hProcess, &exit_code))
        result.exit_code = (int)exit_code;

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (reader.joinable())
        reader.join();
    CloseHandle(out_rd);
    return result;
}
#else
// fork/exec rather than popen, because popen gives no way to stop waiting.
// The Windows branch above honours timeout_sec; a POSIX branch that ignored it
// left the SAME unbounded hang reachable on Linux and macOS while the
// signature advertised otherwise — verified by pointing --sherpa-bin at a
// script that sleeps forever: the process ran until killed externally.
//
// execvp also removes the shell from the path entirely, so a model path
// containing a space or a quote can no longer be re-split or re-interpreted.
inline RunResult run_capture_stdout(const std::vector<std::string>& args, int timeout_sec,
                                    bool capture_stderr = false) {
    RunResult result;
    if (args.empty()) {
        result.spawn_failed = true;
        return result;
    }

    int fds[2];
    if (pipe(fds) != 0) {
        result.spawn_failed = true;
        return result;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        result.spawn_failed = true;
        return result;
    }

    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        if (capture_stderr) {
            dup2(fds[1], STDERR_FILENO);
        } else {
            const int nul = open("/dev/null", O_WRONLY);
            if (nul >= 0) {
                dup2(nul, STDERR_FILENO);
                close(nul);
            }
        }
        close(fds[1]);
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const auto& a : args)
            argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127); // exec failed; 127 is the shell's convention for not-found
    }

    close(fds[1]);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec > 0 ? timeout_sec : 0);
    char buf[4096];
    for (;;) {
        int wait_ms = -1;
        if (timeout_sec > 0) {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
                    .count();
            if (left <= 0) {
                result.timed_out = true;
                break;
            }
            wait_ms = (int)left;
        }
        struct pollfd pfd {
            fds[0], POLLIN, 0
        };
        const int pr = poll(&pfd, 1, wait_ms);
        if (pr == 0) {
            result.timed_out = true;
            break;
        }
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n > 0)
            result.output.append(buf, buf + n);
        else if (n == 0)
            break; // child closed stdout
        else if (errno != EINTR)
            break;
    }
    close(fds[0]);

    if (result.timed_out) {
        kill(pid, SIGKILL);
        // Reap regardless, so a timed-out sherpa cannot become a zombie.
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}
#endif

} // namespace crispasr_cli_process
