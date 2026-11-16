using System;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// The skip policy for tests that need the native library or a real model.
    /// <para>
    /// Issue #291 asked "which tests use a real world gguf model?" — the honest
    /// answer used to be "none in CI": every live test began
    /// <c>if (!CanLoadLibrary()) return;</c>, so a suite that loaded no library
    /// and opened no model was indistinguishable from a passing one. A gate that
    /// cannot go red is not a gate.
    /// </para>
    /// <para>
    /// So: a skip is still a skip locally, but
    /// <c>CRISPASR_CS_REQUIRE_LIVE=1</c> — which CI sets once it has built the
    /// library and downloaded a model — turns every silent skip into a failure.
    /// And a model env var that is set but points at a missing file fails
    /// always, in every configuration: that is a misconfiguration, never a
    /// legitimate skip.
    /// </para>
    /// </summary>
    internal static class Live
    {
        /// <summary>CI has provisioned a library and a model; skips are failures.</summary>
        internal static bool Required
        {
            get
            {
                var v = Environment.GetEnvironmentVariable("CRISPASR_CS_REQUIRE_LIVE");
                return v == "1" || string.Equals(v, "true", StringComparison.OrdinalIgnoreCase);
            }
        }

        /// <summary>
        /// True when the native library loaded. Under
        /// <c>CRISPASR_CS_REQUIRE_LIVE</c> a failure to load throws, and the
        /// message carries the resolver's full probe list.
        /// </summary>
        internal static bool LibraryLoadable()
        {
            try
            {
                _ = Session.AvailableBackends();
                return true;
            }
            catch (DllNotFoundException e) { return NoLibrary("native library did not load: " + e.Message); }
            catch (EntryPointNotFoundException e) { return NoLibrary("native library loaded but is missing exports: " + e.Message); }
        }

        /// <summary>
        /// True when <paramref name="envVar"/> names a model file that exists.
        /// Set-but-missing always throws; unset throws only when
        /// <paramref name="required"/> and <see cref="Required"/> are both true.
        /// </summary>
        internal static bool Model([NotNullWhen(true)] string? path, string envVar, bool required = false)
        {
            if (string.IsNullOrEmpty(path))
                return required && Required
                    ? throw new InvalidOperationException(
                        $"CRISPASR_CS_REQUIRE_LIVE=1 but {envVar} is not set — the real-model tests would all skip silently.")
                    : false;

            if (!File.Exists(path))
                throw new InvalidOperationException(
                    $"{envVar}={path} does not exist. A model path that points nowhere is a misconfiguration, " +
                    "not a reason to skip (issue #291).");

            return true;
        }

        // Only the LIBRARY is unconditionally required under the flag. An
        // optional model that was never provisioned is still a legitimate skip
        // — otherwise enabling the gate at all would demand every model in the
        // suite, and the gate would just get turned back off.
        private static bool NoLibrary(string reason)
        {
            if (Required)
                throw new InvalidOperationException(
                    "CRISPASR_CS_REQUIRE_LIVE=1, but the native library is not usable: " + reason +
                    "\nResolver probed:\n  " +
                    string.Join("\n  ", NativeLibraryResolver.ProbedPaths));
            return false;
        }
    }
}
