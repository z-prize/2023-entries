#pragma once

void* build_constraints_on_gpu(void *wl, void *wr, void *wo, void *w4, cudaStream_t st = 0);