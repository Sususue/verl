import os, pathlib, shutil
root=os.environ.get('DSV4_QAT_ROOT')
if root and os.environ.get("DSV4_CONFIG_ONLY") != "1":
    os.environ.pop('DSA_SCHEME',None)
    root=pathlib.Path(root)
    cache=root/'c/extensions'/str(os.getpid())
    cache.mkdir(parents=True,exist_ok=True)
    seed=root/'c/extension_seed'
    if not (cache/'.ready').exists():
        if seed.is_dir():shutil.copytree(seed,cache,dirs_exist_ok=True,ignore=shutil.ignore_patterns('lock','*.lock','.nfs*'))
        (cache/'.ready').touch()
    os.environ['TORCH_EXTENSIONS_DIR']=str(cache)

    # Every Python process gets its own Triton lock namespace on the task filesystem.
    triton_cache = root/'c/triton_process'/str(os.getpid())
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(triton_cache)
