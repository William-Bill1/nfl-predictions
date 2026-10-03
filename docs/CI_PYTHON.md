# CI Python version

The `tests.yml` workflow runs pytest once on Python 3.14 rather than a
3.12/3.13 matrix. Its pipeline smoke and determinism checks also use 3.14.
`actions/setup-python` resolves the available patch release in that series;
the version is explicit so a future minor release does not silently migrate CI.

This is a CI migration, not a local-environment migration. The existing local
launcher and scheduled data workflows retain their previously configured
versions. Python 3.14 dependency resolution and runtime validation are separate
checks: a successful dependency check alone does not establish test or pipeline
compatibility. Require both CI jobs to pass before merging this change.

The reduction removes one pytest job while retaining the pipeline smoke job.
Broader version support is no longer checked by this workflow.
