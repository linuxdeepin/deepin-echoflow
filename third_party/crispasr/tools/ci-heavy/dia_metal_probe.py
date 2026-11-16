#!/usr/bin/env python3
"""Prove Metal execution is available before spending a model run on macOS."""
import json
import os
from pathlib import Path
import subprocess
import sys
assert sys.platform == 'darwin'
out = Path(os.environ['HEAVY_OUT'])
scratch = Path(os.environ['HEAVY_SCRATCH'])
out.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
subprocess.run(['uptime'], check=True)
subprocess.run(['vm_stat'], check=True)
source = scratch / 'metal-probe.m'
source.write_text(r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
int main(void) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { puts("METAL_UNAVAILABLE: no MTLDevice"); return 77; }
        printf("METAL_DEVICE: %s\n", [[d name] UTF8String]);
        if ([[d name] rangeOfString:@"paravirtual" options:NSCaseInsensitiveSearch].location != NSNotFound) {
            puts("METAL_VIRTUAL_DEVICE_UNSUITABLE: require physical model-execution hardware"); return 78;
        }
        NSError *error = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal; kernel void mark(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }" options:nil error:&error];
        if (!lib) { puts("METAL_SHADER_FAILED"); return 2; }
        id<MTLComputePipelineState> pipeline = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mark"] error:&error];
        if (!pipeline) return 3;
        id<MTLBuffer> buffer = [d newBufferWithLength:4*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [d newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted) return 4;
        const float *result = [buffer contents];
        for (int i=0;i<4;i++) if (result[i] != 1.0f) return 5;
        puts("METAL_EXECUTION_PASS");
        return 0;
    }
}
''')
binary = scratch / 'metal-probe'
subprocess.run(['clang', '-fobjc-arc', '-framework', 'Foundation', '-framework', 'Metal',
                str(source), '-o', str(binary)], check=True, timeout=60)
r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
(out / 'metal-probe.log').write_text(r.stdout + r.stderr)
(out / 'metal-probe.json').write_text(json.dumps({'exit_code':r.returncode, 'output':r.stdout, 'passed':r.returncode==0}, indent=2)+'\n')
print(r.stdout + r.stderr, flush=True)
assert r.returncode == 0, 'No Metal model acceptance can be claimed without executing this GPU probe.'
