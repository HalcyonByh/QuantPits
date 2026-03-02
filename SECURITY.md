# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

We actively maintain the **latest release** on the `main` branch. Older releases will receive security patches on a best-effort basis.

## ⚠️ Important Disclaimer

QuantPits is a **quantitative research and trading system**. Its outputs can directly influence financial decisions. Users are solely responsible for:

- Validating all model outputs and trade signals before executing real trades.
- Securing API keys, brokerage credentials, and any other sensitive information used alongside this system.
- Ensuring that their deployment environment (servers, networks, databases) follows security best practices.

**We strongly recommend never committing credentials, API keys, or personal trading data to any repository.**

## Reporting a Vulnerability

We take security issues seriously. If you discover a vulnerability, please report it responsibly:

1. **Do NOT open a public GitHub Issue.** Security vulnerabilities should not be disclosed publicly until a fix is available.
2. **Email us directly** at **security@quantpits.com** with the subject line: `[SECURITY] QuantPits Vulnerability Report`.
3. Include the following information in your report:
   - A clear description of the vulnerability.
   - Steps to reproduce the issue.
   - The potential impact (e.g., data leakage, code injection, unauthorized access).
   - Any suggested fix or mitigation, if available.

## Response Timeline

| Stage                  | Target SLA     |
| ---------------------- | -------------- |
| Acknowledgment         | 48 hours       |
| Initial assessment     | 5 business days|
| Patch release (if applicable) | 30 days  |

We will keep you informed of our progress and coordinate disclosure timing with you.

## Scope

The following areas are **in scope** for security reports:

- **Data leakage** in model training / prediction pipelines (e.g., look-ahead bias in data splitting).
- **Code injection** via YAML configuration files or user-supplied inputs.
- **Path traversal** in workspace initialization or file I/O operations.
- **Dependency vulnerabilities** in pinned packages listed in `requirements.txt` or `pyproject.toml`.
- **Information disclosure** through logs, MLflow tracking artifacts, or error messages.

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (e.g., Qlib, pandas, NumPy) — please report those to the respective projects.
- Issues arising from user misconfiguration of their own deployment environment.
- Denial-of-service attacks against locally-run Streamlit dashboards.

## Security Best Practices for Users

When deploying QuantPits in a production environment, we recommend:

- **Isolate workspaces**: Use separate OS-level user accounts or containers for each workspace.
- **Protect sensitive data**: Add all workspace `data/`, `output/`, and `mlruns/` directories to `.gitignore` (already configured by default).
- **Pin dependencies**: Use `pip freeze` to lock exact versions in your production environment.
- **Audit configurations**: Review YAML workflow configs before execution, especially if sourced from untrusted origins.
- **Restrict dashboard access**: When running Streamlit dashboards on a network, use a reverse proxy with authentication.

## Acknowledgments

We gratefully acknowledge security researchers and community members who help improve the safety of QuantPits. With your permission, we will credit you in our release notes.
