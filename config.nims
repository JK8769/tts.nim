## Auto-detect backend based on platform.
## Apple Silicon (arm64 macOS) → MLX backend
## Everything else → GGML backend

when defined(macosx) and defined(arm64):
  switch("define", "useMlx")
