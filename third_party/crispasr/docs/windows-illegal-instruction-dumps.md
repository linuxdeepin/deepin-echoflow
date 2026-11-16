# Capturing a Windows illegal-instruction address

Use this when CrispASR exits with `-1073741795` (`0xC000001D`). The exit code
only says that Windows raised `STATUS_ILLEGAL_INSTRUCTION`; the dump identifies
the exact instruction and its offset within `crispasr.exe` or a DLL.
First run the fixed binary normally. If it starts, repeat once from a
Command Prompt with the old eager behavior restored:

```bat
set GGML_CPU_EAGER_UE4M3_LUT=1
rem launch the same crispasr.exe command here
```

Default-starts/eager-crashes is a controlled confirmation that the UE4M3
initialization was the trigger. Clear the variable before testing the default
path again — it persists for the life of that Command Prompt window, so a
"default" run in the same window is still an eager run:

```bat
set GGML_CPU_EAGER_UE4M3_LUT=
```

If the default still crashes, capture its dump as below; a different optimized
CPU path is executing during startup.

## Capture a triage dump

1. Download Microsoft's Sysinternals ProcDump and create `C:\crispasr-dumps`.

2. Open Command Prompt and run this *before* launching CrispASR. **Give the
   final argument as a dump FILE, not a directory** — pointed at a directory,
   ProcDump does not reliably derive a name and you end up with no dump
   (reported in #403):

   ```bat
   procdump64.exe -accepteula -mt -e 1 -w crispasr.exe C:\crispasr-dumps\crispasr_dump
   ```

   That writes `C:\crispasr-dumps\crispasr_dump.dmp` — ProcDump appends the
   extension itself.

   `-mt` writes a triage dump: smaller than the default mini dump, and it strips
   most process memory while keeping the exception context, thread stacks, and
   module list. That is enough for this diagnosis and it is the safer default for
   a file you may end up sharing. If `db @rip` later comes back all `?` marks,
   the bytes at the fault address were not captured — rerun without `-mt` to get
   the default mini dump.

   `-e 1` catches first-chance exceptions as well as unhandled ones. Plain `-e`
   only fires on an *unhandled* exception, and parts of the CUDA and ggml stack
   install their own handlers, so a bare `-e` can sit there and produce nothing.
   The tradeoff is that `-e 1` may also dump an exception the process handles and
   recovers from, so confirm the process actually died before sending anything.

   `procdump64.exe` already captures this 64-bit process correctly; `-64` is
   unnecessary. Add it only if you are running the 32-bit `procdump.exe`.

3. Reproduce the crash once. Do not use `-ma` unless a maintainer asks: a full
   dump may contain model data, audio, paths, and other process memory.

4. Record the exact CrispASR release/filename and the CPU's **instruction-set
   flags**. The flags matter more than the model name, because the usual cause of
   `0xC000001D` is a build compiled for a wider ISA than the CPU supports (#380,
   #374).

   Sysinternals Coreinfo reports them natively, and you are already downloading
   from Sysinternals for ProcDump:

   ```bat
   Coreinfo64.exe -f -accepteula
   ```

   The model name is still worth including:

   ```powershell
   Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
   ```

   From MSYS/Git-Bash, if you prefer:

   ```bash
   grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512[a-z_]*|avx_vnni|fma|f16c|sse4_[12]|ssse3|bmi[12])$' | sort
   ```

   No `avx2` in that list means the AVX2 build cannot run: use the
   `crispasr-windows-x86_64-cpu-legacy` package. From v0.8.30 the CLI should detect
   this itself and say so instead of dying silently.

## If step 2 produced no dump file

Windows Error Reporting can capture the process regardless of how it exits, with
nothing to download. Create this key and reproduce the crash again:

```
HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\crispasr.exe
  DumpFolder  (REG_EXPAND_SZ)  C:\crispasr-dumps
  DumpType    (REG_DWORD)      1     ; 1 = mini, 2 = full
```

Creating the key requires an elevated Command Prompt or Registry Editor. Leave
`DumpType` at `1`; `2` carries the same privacy concerns as ProcDump's `-ma`.
Delete the key once you have the dump.

## Extract the useful address locally

Open the `.dmp` in WinDbg. If `windbg` is not on your `PATH`, note that there are
two builds: the classic `windbg.exe` from Debugging Tools for Windows in the
Windows SDK, and the modern Store app, which installs as **`WinDbgX.exe`** under
`%LOCALAPPDATA%\Microsoft\WindowsApps`. Either works.

```text
.logopen /t crispasr-illegal-instruction.txt
.sympath C:\real\path\to\crispasr\build
.exepath+ C:\real\path\to\crispasr\build
.reload
.ecxr
r
u @rip L10
db @rip L20
lm
k
.logclose
```

Give the folder holding `crispasr.exe` and `ggml-cpu.dll`, not a single DLL.

- `.sympath` has **no** `+`. It replaces the search path rather than appending to
  it. If `_NT_SYMBOL_PATH` is set in your environment — common once the SDK
  debugging tools are installed — it usually already contains
  `srv*https://msdl.microsoft.com/download/symbols`, and appending would inherit
  that. Check with `echo %_NT_SYMBOL_PATH%` if you want to confirm.
- `.exepath` is the setting that matters for a release zip. `.sympath` searches
  for **PDBs**; `.exepath` searches for the **DLL and EXE images** themselves,
  and it is the one that clears the `Unable to load image … Win32 error 0n2`
  failures. Release packages generally ship no PDBs, so `.sympath` pointed at the
  same folder finds nothing (harmless - but a build with
  PDBs alongside the binaries would pick them up).
- `.reload` has no `/f`. Force-loading pulls in every module in the list.

`!analyze -v` (a bucketing wrapper that resolves the
whole module list to build a stack, normally not necessary, the fault
address comes from `.ecxr` and `u @rip`) can be run as a
separate second pass after you already have the address. Two things to know
before trusting its output: pointed at the Microsoft symbol server it downloads
PDBs for every OS module in the list, which is slow and can fail outright
(#403); run offline with the paths above it finishes in about a second but
mis-buckets the crash as `WRONG_SYMBOLS … ntdll.wrong.symbols.dll`. Both are
cosmetic — its `ExceptionCode: c000001d (Illegal instruction)` line is correct
either way, and the fields that matter come from the main sequence.

## What a useful result looks like

The instruction at `RIP` should be a wide vector op — something like
`vpdpbusd`, `vfmadd231ps`, or any `v*` instruction with a `zmm` or `ymm`
operand — and the module containing `RIP` should be `ggml-cpu.dll`. That is the
ISA-mismatch signature. A ZMM operand on a host whose flags show only AVX2/FMA
is the confirmed shape of #374.

Once `.exepath` finds the DLL, the disassembly labels the address for you —
WinDbg substitutes an underscore for the hyphen in the module name. With no
PDBs the label comes from the DLL's export table, so it usually reads as the
nearest exported function plus an offset (`ggml_cpu!ggml_graph_compute+0x…`),
and `lm` shows the module as `(export symbols)`. If the image was not mapped,
the label degrades to the bare `ggml_cpu+0x98e9` form instead. Either way the
log carries the module load address from `lm`, so there is no need to subtract
the load base by hand.

`db @rip L20` is there because the raw bytes still identify the instruction when
the disassembler chokes on an encoding it does not know.

If `RIP` lands in `nvcuda.dll` or another driver module instead, this is not an
ISA mismatch and the triage goes elsewhere: report the driver version alongside
the log.

If `.ecxr` reports that no context is available, the dump was captured at the
wrong moment. That is a capture problem, not a symbol one — go back to the
ProcDump step.

## Before you post

Scrub the log first. `.exepath`, `lm`, and `k` all emit local paths, so the file
will usually contain your Windows username and may contain model or audio paths.
Skim it and redact before attaching.

Then send `crispasr-illegal-instruction*.txt`, the CPU name and flags, and the
exact binary — not the dump — in the public issue. The key fields are the
exception address, the module containing `RIP`, the disassembled instruction at
`RIP`, and the module load address. Those let maintainers calculate an
ASLR-independent module offset and resolve it against the matching
executable/PDB.

If the text is insufficient, share the `.dmp` privately. Even a triage dump can
contain local paths, so do not attach it publicly without reviewing that risk.
