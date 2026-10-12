# engine-data

Written by the engine container (mounted as `/work`): `engine.log` and `cache/` (Triton kernels, vLLM and Hugging Face
caches, the compiled PLE NVMe reader). Everything here except this file is git-ignored and safe to delete while the
engine is stopped; it is rebuilt at the next start (which then takes a little longer). Put it elsewhere with
`ENGINE_DATA=/path` in `.env`.
