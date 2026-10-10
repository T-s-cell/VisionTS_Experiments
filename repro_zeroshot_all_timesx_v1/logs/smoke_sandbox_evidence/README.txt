Sandbox plumbing smoke (2026-10-10, /tmp/vz_smoke):
- stub reuse_final.json (all 190 vars reuse) + zero-valued cache_all for the 1,693
  fill windows (no GPU locally); pipeline chain assemble->evaluate->recompute->
  verify_acceptance ran end to end: 8,106 windows (cache=6,413, new=1,693),
  independent recompute PASS (dev 0), 16/16 acceptance checks PASS.
- subset_1677_purecache_reproduction.json is REAL evidence: computed from the
  genuine old cache npz only — 3.254715/1.042067, delta 9.18e-08/3.04e-07 <= 5e-7.
- stub_generator.py is the sandbox fabricator; it is never used by the real run.
