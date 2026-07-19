# AI and development-tool disclosure template

Copy this file into the submission evidence package and complete it through
human review. It is a working provenance record, not a claim of legal or rules
compliance. Re-check the current official rules and disclosure format before
submitting.

## Submission identity

- Team:
- Submission/repository commit:
- Dirty-diff hash, if any:
- Simulator build and mode:
- Environment lock and dependency-inventory file:
- Review date:
- Human reviewer:

## Generative-AI tools used

Add one row per materially used tool or model. Do not include secrets,
credentials, private captures, or full prompts containing restricted data.

| Provider/tool | Model/version or surface | Dates used | Purpose | Material outputs retained | Human verification |
|---|---|---|---|---|---|
| | | | | | |

## Other material development tools and services

| Tool/service | Version | Purpose | Output or dependency introduced | Human verification |
|---|---|---|---|---|
| | | | | |

## Open-source dependency review

- CycloneDX inventory path and SHA-256:
- Runtime lock path and SHA-256:
- Dependencies added outside the lock:
- License/source-notice review completed by:
- Required notices bundled at:
- Known exceptions or unresolved questions:

## Attestation notes

- Describe the tests and manual review applied to generated code or assets.
- Identify any generated output that was rejected or substantially rewritten.
- Confirm private `captures/` data and credentials are absent from the
  submission repository.
- Record any difference between public qualifier documentation and the
  empirically verified build-specific interface used by the code.
