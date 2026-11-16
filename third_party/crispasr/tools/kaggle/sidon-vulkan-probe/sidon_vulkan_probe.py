#!/usr/bin/env python3
"""
#416 feasibility probe — 3 questions, no build, ~2 minutes.

Before committing a ~20 min CrispASR+Vulkan build to a Kaggle kernel, establish
that the run could produce a positive at all. Three things must ALL hold, and
each has killed an experiment in this issue already:

  Q1 WHICH GPU?  T4 is sm_75 (dp4a -> integerDotProduct...Accelerated TRUE, the
     same compute capability as the reporter's GTX 1660 SUPER). P100 is sm_60:
     no dp4a, so ggml disables MMQ and the arm under test cannot run. Six
     consecutive P100 draws on $KAGGLE_ACCOUNT today, and no working API-side selector,
     so this is a lottery — fail fast and report the draw.
  Q2 IS VULKAN EVEN PRESENT?  Kaggle images are not general-purpose. The NVIDIA
     driver normally ships an ICD but nobody has confirmed it here. No ICD means
     the entire Vulkan plan is void on Kaggle, which is a legitimate negative.
  Q3 IS int-dot ACCELERATED?  The number that actually decides whether MMQ
     pipelines are built and dispatched. Reported natively rather than forced is
     the whole reason to prefer this over lavapipe.

Writes probe.json and exits 0 either way — a "no" here is a real answer.
"""
import json
import subprocess
from pathlib import Path

OUT = Path("/kaggle/working/probe.json")


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


res = {}

# Q1 — the draw. Cheapest, and decides whether anything else matters.
gpu = sh("nvidia-smi -L").stdout.strip()
cap = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader").stdout.strip().splitlines()
cc = cap[0].strip() if cap else ""
res["gpu"] = gpu
res["compute_cap"] = cc
try:
    res["dp4a_capable"] = float(cc) >= 6.1
except ValueError:
    res["dp4a_capable"] = False
# sm_75 == the reporter's hardware class, so int-dot should be ACCELERATED, not forced.
res["reporter_class_sm75"] = cc.startswith("7.5")
print(f"Q1 gpu={gpu!r} compute_cap={cc!r} dp4a={res['dp4a_capable']} sm75={res['reporter_class_sm75']}",
      flush=True)

# Q2 — does a Vulkan ICD exist for the NVIDIA driver?
sh("apt-get update -qq", timeout=900)
sh("apt-get install -y -qq vulkan-tools libvulkan1 mesa-vulkan-drivers", timeout=1800)
icds = sh("ls /usr/share/vulkan/icd.d/ 2>/dev/null").stdout.split()
res["vulkan_icds"] = icds
vi = sh("vulkaninfo --summary 2>&1 | head -60")
res["vulkaninfo_rc"] = vi.returncode
res["vulkaninfo_head"] = (vi.stdout or "")[:4000]
devs = [ln.strip() for ln in (vi.stdout or "").splitlines() if "deviceName" in ln]
res["vulkan_devices"] = devs
res["nvidia_vulkan_present"] = any("NVIDIA" in d or "Tesla" in d or "T4" in d for d in devs)
print(f"Q2 icds={icds} devices={devs}", flush=True)

# Q3 — the accelerated bit, read from the device properties.
acc = sh("vulkaninfo 2>/dev/null | grep -i integerDotProduct4x8BitPackedSignedAccelerated | head -2")
res["int_dot_accelerated_lines"] = (acc.stdout or "").strip().splitlines()
res["int_dot_accelerated"] = "true" in (acc.stdout or "").lower()
print(f"Q3 int_dot_accelerated={res['int_dot_accelerated']} raw={res['int_dot_accelerated_lines']}", flush=True)

# Q4 — if there is no NVIDIA ICD, can one be installed? This decides whether the
# Kaggle-Vulkan route is DEAD or merely blocked on the T4 lottery. The driver is
# present (nvidia-smi works); what is missing is the Vulkan loader's ICD manifest,
# normally shipped by libnvidia-gl-<ver>.
if not res["nvidia_vulkan_present"]:
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip().splitlines()
    drv = drv[0].split(".")[0] if drv else ""
    res["driver_major"] = drv
    # Is the manifest already on disk somewhere the loader doesn't scan?
    found = sh("find / -name 'nvidia_icd*.json' -not -path '*/proc/*' 2>/dev/null | head -5").stdout.split()
    res["nvidia_icd_on_disk"] = found
    inst = sh(f"apt-get install -y -qq libnvidia-gl-{drv} 2>&1 | tail -5", timeout=1800)
    res["icd_install_tail"] = (inst.stdout or "")[-1500:]
    icds2 = sh("ls /usr/share/vulkan/icd.d/ 2>/dev/null").stdout.split()
    res["vulkan_icds_after_install"] = icds2
    vi2 = sh("vulkaninfo --summary 2>&1 | head -40")
    devs2 = [ln.strip() for ln in (vi2.stdout or "").splitlines() if "deviceName" in ln]
    res["vulkan_devices_after_install"] = devs2
    res["nvidia_vulkan_after_install"] = any("NVIDIA" in d or "Tesla" in d for d in devs2)
    print(f"Q4 driver={drv} icd_on_disk={found} after_install={devs2}", flush=True)

res["verdict"] = {
    "worth_building": bool(res["nvidia_vulkan_present"] and res["dp4a_capable"]),
    "native_not_forced": bool(res["int_dot_accelerated"]),
    "reason": (
        "P100 draw — sm_60 has no dp4a, MMQ disabled, arm under test cannot run"
        if not res["dp4a_capable"] else
        "no NVIDIA Vulkan ICD on this image — Vulkan plan is void on Kaggle"
        if not res["nvidia_vulkan_present"] else
        "GO: NVIDIA Vulkan present on dp4a-capable hardware"
    ),
}
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res["verdict"], indent=2), flush=True)
