# Project instructions

- Common structure guide: [WORKFLOW_GUIDE.md](https://github.com/berandaNuguri/berandaNuguri/blob/main/WORKFLOW_GUIDE.md), version 1.5.
- Read README.md and inspect affected imports, entry points and resource paths before changing paths.
- Keep all executable source under src/jpeg_aware_steganalysis/ and entry points under scripts/.
- Preserve model computations, preprocessing, splits, random seeds, CLI defaults, and state_dict keys.
- The original paper project is outside this checkout. Do not modify it as part of this copy.
- Do not add compatibility symlinks or legacy top-level arch/ and utils/ packages.
- Exclude datasets, CSV files, figures, checkpoints, results, logs, and manuscripts from Git.
- Read project-specific instructions before verification; report only checks actually performed.
- README content policy: follow the paper-focused README section of WORKFLOW_GUIDE v1.4. Lead with the research problem, verified paper metadata and method; keep migration history and verification logs in docs/ or AGENTS.md.
- Verify publication metadata from the paper or primary publication record. Do not infer acceptance, venue or year from a directory name or manuscript template.
- Applied data/upload policy from WORKFLOW_GUIDE v1.5: actual datasets belong in ignored src/data/ or external storage; executable preparation/loading code belongs in utils/data_processing/.
- W&B local runs, account settings, binary histories and credentials must never be committed. Authenticate through the shell or a private external credential store.

- Resolve task CLI paths from this checkout, independent of the calling shell directory. Keep actual datasets under ignored src/data/ and model files under ignored checkpoints/ by default. RESEARCH_DATA_ROOT and RESEARCH_MODEL_ROOT may select private external storage. Historical input-prefix mappings belong in the private RESEARCH_PATH_MAP JSON environment variable; never hardcode host paths or container names.
