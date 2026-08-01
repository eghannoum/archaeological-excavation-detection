# Security Policy

## Supported versions

Security fixes are applied to the latest minor release. Only the most recent
release line receives security updates.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Report
them privately so they can be fixed before disclosure:

- **Preferred:** use the GitHub private vulnerability reporting feature on this
  repository (`Security` tab > "Report a vulnerability"). This lets you submit a
  report that is visible only to the maintainers, including optional draft
  advisories and proposed fixes.
- **Alternative:** if private vulnerability reporting is unavailable for this
  repository, contact the maintainers through a private message or a channel
  listed in the repository description, describing the issue without
  publishing exploit details.

Please include in your report:

- The affected component, script, or configuration.
- Steps to reproduce the issue (minimal example preferred).
- The impact you observed and any suggested mitigation.

You will receive an acknowledgement within 7 days and an assessment of the
report, including an indication of when a fix may be expected. We ask that you
keep the details confidential until a fix has been released and coordinated
disclosure has occurred.

## Scope

This policy covers the code in this repository. The full dataset and trained
model weights are not distributed with the repository; reports about model
behavior (e.g. bias, robustness) are welcome through the same private channels
but are handled as research feedback rather than security fixes.
