# Qwen serving memory reservation

Qwen serving allocates model weights, the full configured context, and a
bounded decoder workspace before reporting ready. It runs one maximum-size
prefill and one decode step during loading, resets the context, then seals
the scratch pools. Prefill batches use at most 2048 rows (or max_seq for
smaller contexts). Smaller requests borrow logical prefixes of those same
buffers instead of retaining new allocations for each prompt length.

Image embedding prefill uses the same bounded batches and reserved position
storage. When vision is enabled, loading also encodes a synthetic image at
the configured maximum image area, replacing libmtmd's smaller default
warmup. The bridge and GPU driver still own their internal allocations;
decoder pool sealing does not forbid allocations inside those libraries.
GPU validation of maximum vision shapes remains necessary before deployment.

If initial reservation fails, the model never becomes ready. Reduce context
capacity or use CPU vision rather than discovering insufficient decoder
workspace during a request. Error cleanup retains sealed decoder buffers.
Per-size prefill graph capture is disabled in reserved serving mode to avoid
creating new graph allocations as prompt sizes change.

The dashboard's VRAM bar reports allocated memory. Context positions report
occupied physical rows separately, including during Qwen decode. Scratch
borrowed now and peak count logical prefixes checked out of pools; they
are not measurements of kernel execution residency. The peak includes
startup warmup, and idle borrowed scratch is zero even while reserved VRAM
stays allocated.

Gemma serving and standalone CLI runtimes retain their existing allocation
policies. No server configuration change is required for Qwen reservation.
