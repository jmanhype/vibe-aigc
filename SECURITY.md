# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Vibe AIGC, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Best Practices

When using Vibe AIGC:

- **API Keys**: Never commit API keys to version control. Use environment variables.
- **Checkpoints**: Checkpoint files may contain sensitive workflow data. Secure accordingly.
- **LLM Outputs**: Validate and sanitize any LLM-generated content before use in production.
- **Dependencies**: Keep dependencies updated to patch known vulnerabilities.

## Scope

This policy covers the `vibe-aigc` Python package. Third-party dependencies have their own security policies.
