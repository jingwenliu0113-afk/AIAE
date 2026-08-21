# Security Policy

## Supported version

BrickAgain is currently a research prototype without stable releases. Security
fixes apply to the current default branch only.

## Reporting a vulnerability

Do not include credentials, private data, gated model artifacts, or working
exploit details in a public issue. Use GitHub's private vulnerability reporting
for the repository when it is enabled. If that channel is not yet available,
contact the maintainer through the GitHub profile and request a private contact
method before sending sensitive details.

Please include:

- the affected file or component;
- the impact and conditions required to reproduce it;
- minimal reproduction steps that do not contain private data or secrets; and
- any suggested mitigation.

## Credential and artifact handling

Hugging Face tokens and other credentials must remain outside the repository.
Raw/processed datasets, gated model weights, BrickGPT weights, and locally
trained checkpoints must not be attached to issues or pull requests unless
their redistribution terms have been reviewed explicitly.
