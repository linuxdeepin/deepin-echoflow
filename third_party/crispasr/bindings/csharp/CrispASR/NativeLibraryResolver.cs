using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace CrispASR
{
    /// <summary>
    /// Locates the native crispasr shared library and loads it by explicit
    /// path, instead of leaving it to the runtime's bare-name probing.
    /// <para>
    /// Issue #291: the managed assembly used to be named <c>CrispASR.dll</c>
    /// while the native library is <c>crispasr.dll</c> — on Windows those are
    /// the SAME file name, so the two could not live in one directory and
    /// bare-name probing found the managed assembly instead of the native
    /// library. The assembly is now <c>CrispASR.Net.dll</c> (the namespace and
    /// every type name are unchanged), and this resolver additionally refuses
    /// to hand a managed assembly to the native loader, so the failure mode can
    /// never come back silently.
    /// </para>
    /// <para>
    /// Search order — first hit wins, and every candidate is loaded by full
    /// path so its sibling <c>ggml*.dll</c> / <c>libggml*.so</c> dependencies
    /// resolve out of the same directory:
    /// <list type="number">
    /// <item><c>CRISPASR_LIBRARY_PATH</c> — the library file itself, or a directory holding it</item>
    /// <item><c>runtimes/&lt;rid&gt;/native/</c> under the app base (the NuGet layout)</item>
    /// <item><c>native/</c>, <c>bin/</c>, <c>lib/</c> under the app base (the release-tarball layout)</item>
    /// <item>the app base directory itself</item>
    /// <item>the directory holding this assembly, then the current directory, each with the same sub-layouts</item>
    /// <item>the runtime's own default probing, as a last resort</item>
    /// </list>
    /// </para>
    /// </summary>
    public static class NativeLibraryResolver
    {
        /// <summary>The name every <c>[DllImport]</c> in this binding uses.</summary>
        internal const string LibraryName = "crispasr";

        private static readonly object Gate = new object();
        private static bool _installed;
        private static string? _resolvedPath;
        private static string[] _probedPaths = Array.Empty<string>();
        private static string[] _skippedManaged = Array.Empty<string>();

        /// <summary>
        /// Installed by <c>NativeMethods</c>'s type initializer, which the CLR
        /// runs before the first P/Invoke through it. Calling it again is a
        /// no-op; it is public so a host that sets <c>CRISPASR_LIBRARY_PATH</c>
        /// programmatically can force registration at a known point.
        /// </summary>
        public static void Install()
        {
            lock (Gate)
            {
                if (_installed) return;
                _installed = true;
                NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);
            }
        }

        /// <summary>
        /// Full path of the native library that was loaded, or <c>null</c>
        /// before the first P/Invoke (or when the runtime's default probing
        /// found it without help from this resolver).
        /// </summary>
        public static string? ResolvedPath => _resolvedPath;

        /// <summary>Every path examined during the last resolve attempt, in order.</summary>
        public static IReadOnlyList<string> ProbedPaths => _probedPaths;

        /// <summary>The platform file name of the native library.</summary>
        public static string NativeFileName =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "crispasr.dll" :
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libcrispasr.dylib" :
                                                              "libcrispasr.so";

        /// <summary>
        /// Force the load now and return the path it came from. Throws
        /// <see cref="DllNotFoundException"/> with the full probe list when the
        /// library cannot be found — the diagnostic to print when a binding
        /// "does nothing" on a machine where the CLI works.
        /// </summary>
        public static string Load()
        {
            Install();
            var handle = Resolve(LibraryName, typeof(NativeLibraryResolver).Assembly, null);
            if (handle == IntPtr.Zero)
                throw new DllNotFoundException(BuildMessage());
            return _resolvedPath ?? NativeFileName;
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
                return IntPtr.Zero;

            var cached = _resolvedPath;
            if (cached != null && NativeLibrary.TryLoad(cached, out var reuse))
                return reuse;

            var probed = new List<string>();
            var skipped = new List<string>();

            foreach (var candidate in Candidates())
            {
                probed.Add(candidate);
                if (!File.Exists(candidate)) continue;
                if (IsManagedAssembly(candidate)) { skipped.Add(candidate); continue; }
                if (NativeLibrary.TryLoad(candidate, out var handle))
                {
                    _resolvedPath = candidate;
                    _probedPaths = probed.ToArray();
                    _skippedManaged = skipped.ToArray();
                    return handle;
                }
            }

            _probedPaths = probed.ToArray();
            _skippedManaged = skipped.ToArray();

            // Last resort: whatever the runtime's own probing can find (a
            // system-wide install, LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, ...).
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out var fallback))
                return fallback;

            throw new DllNotFoundException(BuildMessage());
        }

        /// <summary>Test hook: the probe list, materialised without loading anything.</summary>
        internal static string[] CandidatePaths() => new List<string>(Candidates()).ToArray();

        private static IEnumerable<string> Candidates()
        {
            var file = NativeFileName;
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            var pinned = Environment.GetEnvironmentVariable("CRISPASR_LIBRARY_PATH");
            if (!string.IsNullOrEmpty(pinned))
            {
                if (Directory.Exists(pinned))
                {
                    foreach (var c in InDirectory(pinned, file))
                        if (seen.Add(c)) yield return c;
                }
                else if (seen.Add(pinned))
                {
                    yield return pinned;
                }
            }

            foreach (var root in Roots())
                foreach (var c in InDirectory(root, file))
                    if (seen.Add(c)) yield return c;
        }

        private static IEnumerable<string> Roots()
        {
            var appBase = AppContext.BaseDirectory;
            if (!string.IsNullOrEmpty(appBase)) yield return appBase;

            var asmDir = Path.GetDirectoryName(typeof(NativeLibraryResolver).Assembly.Location);
            if (!string.IsNullOrEmpty(asmDir)) yield return asmDir!;

            var cwd = Environment.CurrentDirectory;
            if (!string.IsNullOrEmpty(cwd)) yield return cwd;
        }

        private static IEnumerable<string> InDirectory(string dir, string file)
        {
            // The NuGet runtime-asset layout first, so a package always wins
            // over a stray copy someone dropped beside the app.
            string rid;
            try { rid = RuntimeInformation.RuntimeIdentifier; }
            catch { rid = ""; }
            if (!string.IsNullOrEmpty(rid))
                yield return Path.Combine(dir, "runtimes", rid, "native", file);

            // The layout of the libcrispasr-<platform>.tar.gz release archives.
            yield return Path.Combine(dir, "native", file);
            yield return Path.Combine(dir, "bin", file);
            yield return Path.Combine(dir, "lib", file);
            yield return Path.Combine(dir, file);
        }

        // A managed assembly is a perfectly valid PE file, so LoadLibrary can
        // "succeed" on one and then fail at the first entry-point lookup with a
        // message that names the export, not the mistake. GetAssemblyName is the
        // cheap authoritative test: it throws BadImageFormatException on a
        // native library and succeeds on a managed one.
        // internal, not private: tests/test hooks assert the #291 guard directly
        // rather than needing a Windows machine to reproduce the collision.
        internal static bool IsManagedAssembly(string path)
        {
            try
            {
                AssemblyName.GetAssemblyName(path);
                return true;
            }
            catch (BadImageFormatException) { return false; }
            catch (FileLoadException) { return true; }  // already loaded => managed
            catch { return false; }
        }

        private static string BuildMessage()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("CrispASR: could not load the native library '")
              .Append(NativeFileName).Append("'.").AppendLine();

            if (_skippedManaged.Length > 0)
            {
                sb.AppendLine("A MANAGED assembly was found where the native library was expected — skipped:");
                foreach (var p in _skippedManaged) sb.Append("  ! ").AppendLine(p);
                sb.AppendLine("  (rename or move it; the native library and a managed assembly cannot");
                sb.AppendLine("   share a file name on Windows, which is case-insensitive.)");
            }

            sb.AppendLine("Probed, in order:");
            foreach (var p in _probedPaths) sb.Append("  - ").AppendLine(p);

            sb.AppendLine("Fixes:");
            sb.AppendLine("  * set CRISPASR_LIBRARY_PATH to the library file, or to the directory holding it;");
            sb.Append("  * or copy ").Append(NativeFileName)
              .AppendLine(" (with its sibling ggml libraries) next to your application;");
            sb.AppendLine("  * downloads: https://github.com/CrispStrobe/CrispASR/releases");
            sb.AppendLine("    (libcrispasr-windows-x86_64.tar.gz, libcrispasr-linux-x86_64.tar.gz,");
            sb.AppendLine("     libcrispasr-macos-arm64.tar.gz, ... — the bin/ directory of the archive).");
            return sb.ToString();
        }
    }
}
