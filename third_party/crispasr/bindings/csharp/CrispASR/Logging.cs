using System;

namespace CrispASR
{
    /// <summary>Native log severity, matching ggml_log_level.</summary>
    public enum LogLevel
    {
        None = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
    }

    /// <summary>
    /// Controls CrispASR's native log output (issue #291 asked how to reduce it).
    ///
    /// By default the native library prints info-level chatter to stderr. Call
    /// <see cref="SetMinLevel"/> to raise the threshold, <see cref="Silence"/> to
    /// mute it entirely, or <see cref="SetCallback"/> to route every line into
    /// your own logger.
    /// </summary>
    public static class Logging
    {
        // The native side keeps a raw function pointer to whatever we hand it.
        // If the managed delegate is collected, that pointer dangles and the next
        // native log call jumps into freed memory. So the delegate MUST be held in
        // a static field for the process lifetime — this is the single most common
        // P/Invoke-callback crash, and the reason this is not a local variable.
        private static NativeMethods.GgmlLogCallback? _pinned;
        private static Action<LogLevel, string>? _userSink;
        private static LogLevel _minLevel = LogLevel.Debug;
        private static readonly object _gate = new object();

        /// <summary>
        /// Suppress native log lines below <paramref name="min"/>. E.g.
        /// <c>SetMinLevel(LogLevel.Warn)</c> keeps warnings and errors, drops info/debug.
        /// </summary>
        public static void SetMinLevel(LogLevel min)
        {
            lock (_gate)
            {
                _minLevel = min;
                _userSink = null;
                Install();
            }
        }

        /// <summary>Mute all native log output.</summary>
        public static void Silence() => SetMinLevel(LogLevel.None);

        /// <summary>
        /// Route every native log line to <paramref name="sink"/> instead of stderr.
        /// The sink is invoked from a native thread — keep it cheap and thread-safe.
        /// Pass <c>null</c> to restore default stderr logging.
        /// </summary>
        public static void SetCallback(Action<LogLevel, string>? sink)
        {
            lock (_gate)
            {
                _userSink = sink;
                if (sink == null)
                {
                    // Restore the library default by clearing our callback.
                    _pinned = null;
                    NativeMethods.whisper_log_set(null, IntPtr.Zero);
                    return;
                }
                _minLevel = LogLevel.Debug; // sink decides; don't pre-filter
                Install();
            }
        }

        private static void Install()
        {
            // Rebuild and re-pin on every change so the captured _minLevel/_userSink
            // are the current ones; the old delegate stays referenced until this
            // assignment, then becomes collectable safely (native now points here).
            _pinned = (int level, string? text, IntPtr _) =>
            {
                var sink = _userSink;
                if (sink != null)
                {
                    sink((LogLevel)level, text ?? string.Empty);
                    return;
                }
                if (_minLevel == LogLevel.None) return;                  // fully muted
                // GGML_LOG_LEVEL_CONT (5) continues the previous line — always let
                // it through so a multi-line message that opened above threshold is
                // not chopped mid-way.
                if (level != 5 && level < (int)_minLevel) return;
                Console.Error.Write(text);
            };
            NativeMethods.whisper_log_set(_pinned, IntPtr.Zero);
        }
    }
}
