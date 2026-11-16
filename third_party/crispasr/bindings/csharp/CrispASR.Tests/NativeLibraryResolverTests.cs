using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using CrispASR;
using Xunit;

namespace CrispASR.Tests
{
    /// <summary>
    /// Issue #291 (GeorgeS2019): on Windows the managed assembly was
    /// <c>CrispASR.dll</c> and the native library is <c>crispasr.dll</c> — the
    /// same file name on a case-insensitive filesystem. The two could not be
    /// installed side by side, and bare-name P/Invoke probing found the managed
    /// assembly. These tests pin both halves of the fix on every platform, so
    /// the regression does not need a Windows machine to be caught.
    /// </summary>
    public class NativeLibraryResolverTests
    {
        [Fact]
        public void ManagedAssemblyIsNeverOfferedToTheNativeLoader()
        {
            // This test assembly and the binding assembly are both managed;
            // handing either to LoadLibrary is the #291 failure mode.
            var managed = typeof(Session).Assembly.Location;
            Assert.False(string.IsNullOrEmpty(managed), "binding assembly has no on-disk location");
            Assert.True(NativeLibraryResolver.IsManagedAssembly(managed),
                $"{managed} is a managed assembly and must be recognised as one");
        }

        [Fact]
        public void ANonPeFileIsNotMistakenForAManagedAssembly()
        {
            var tmp = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            File.WriteAllBytes(tmp, new byte[] { 0x7f, (byte)'E', (byte)'L', (byte)'F', 0, 1, 2, 3 });
            try
            {
                Assert.False(NativeLibraryResolver.IsManagedAssembly(tmp));
            }
            finally { File.Delete(tmp); }
        }

        // The managed assembly must not be named so that its file collides with
        // the native library on a case-insensitive filesystem. Comparing the two
        // names with OrdinalIgnoreCase is exactly the Windows filesystem rule.
        [Fact]
        public void ManagedAssemblyFileNameDoesNotCollideWithTheNativeLibrary()
        {
            var managedFile = Path.GetFileName(typeof(Session).Assembly.Location);
            foreach (var nativeFile in new[] { "crispasr.dll", "libcrispasr.so", "libcrispasr.dylib" })
                Assert.False(string.Equals(managedFile, nativeFile, StringComparison.OrdinalIgnoreCase),
                    $"managed assembly '{managedFile}' collides with native '{nativeFile}' (issue #291)");
        }

        [Fact]
        public void NativeFileNameMatchesThePlatform()
        {
            var expected =
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "crispasr.dll" :
                RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libcrispasr.dylib" :
                                                                  "libcrispasr.so";
            Assert.Equal(expected, NativeLibraryResolver.NativeFileName);
        }

        [Fact]
        public void ProbeListCoversTheReleaseArchiveLayouts()
        {
            var candidates = NativeLibraryResolver.CandidatePaths();
            Assert.NotEmpty(candidates);

            var file = NativeLibraryResolver.NativeFileName;
            // Every candidate must end in the platform library name...
            Assert.All(candidates, c => Assert.Equal(file, Path.GetFileName(c)));

            // ...and the layouts a user actually gets must all be probed:
            // the libcrispasr-<platform> archive ships the library in bin/, a
            // NuGet package ships it under runtimes/<rid>/native/, and a manual
            // copy lands flat next to the app.
            var appBase = AppContext.BaseDirectory;
            Assert.Contains(candidates, c => c == Path.Combine(appBase, "bin", file));
            Assert.Contains(candidates, c => c == Path.Combine(appBase, "native", file));
            Assert.Contains(candidates, c => c == Path.Combine(appBase, file));
            Assert.Contains(candidates, c => c.Contains(Path.Combine("runtimes")) && c.Contains("native"));
        }

        [Fact]
        public void ProbeListHasNoDuplicates()
        {
            var candidates = NativeLibraryResolver.CandidatePaths();
            Assert.Equal(candidates.Length,
                         candidates.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        }

        // CRISPASR_LIBRARY_PATH is the documented escape hatch for "the CLI
        // finds it, the binding does not". Point it at a file and that exact
        // file must be the first thing tried.
        [Fact]
        public void PinnedLibraryPathIsProbedFirst()
        {
            const string key = "CRISPASR_LIBRARY_PATH";
            var saved = Environment.GetEnvironmentVariable(key);
            var pinned = Path.Combine(Path.GetTempPath(), "pinned-" + NativeLibraryResolver.NativeFileName);
            try
            {
                Environment.SetEnvironmentVariable(key, pinned);
                Assert.Equal(pinned, NativeLibraryResolver.CandidatePaths().First());
            }
            finally { Environment.SetEnvironmentVariable(key, saved); }
        }

        [Fact]
        public void PinnedDirectoryIsExpandedIntoTheKnownLayouts()
        {
            const string key = "CRISPASR_LIBRARY_PATH";
            var saved = Environment.GetEnvironmentVariable(key);
            var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            Directory.CreateDirectory(dir);
            try
            {
                Environment.SetEnvironmentVariable(key, dir);
                var candidates = NativeLibraryResolver.CandidatePaths();
                var file = NativeLibraryResolver.NativeFileName;
                Assert.Contains(candidates, c => c == Path.Combine(dir, file));
                Assert.Contains(candidates, c => c == Path.Combine(dir, "bin", file));
            }
            finally
            {
                Environment.SetEnvironmentVariable(key, saved);
                Directory.Delete(dir, recursive: true);
            }
        }
    }
}
