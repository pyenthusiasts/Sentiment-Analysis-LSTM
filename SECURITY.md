# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Sentiment Analysis LSTM seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: [your.email@example.com]

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include in Your Report

Please include the following information:

- Type of vulnerability (e.g., XSS, SQLi, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- We will acknowledge receipt of your vulnerability report
- We will send you a more detailed response indicating the next steps
- We will work with you to understand and resolve the issue
- We will keep you informed about our progress
- We will credit you in the security advisory (if you wish)

## Security Best Practices

When using this package in production:

### 1. Model Security
- Store trained models in secure locations
- Use access controls for model files
- Validate model integrity before loading

### 2. Input Validation
- Always validate user input before processing
- Set appropriate text length limits
- Sanitize inputs to prevent injection attacks

### 3. API Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication and authorization
- Configure CORS appropriately
- Use environment variables for sensitive configuration

### 4. Dependency Security
- Regularly update dependencies
- Use `safety` to check for known vulnerabilities
- Pin dependency versions in production

### 5. Data Privacy
- Don't log sensitive user data
- Implement proper data retention policies
- Comply with GDPR/CCPA if applicable

### 6. Docker Security
- Don't run containers as root
- Use minimal base images
- Scan images for vulnerabilities
- Keep base images updated

## Security Checklist for Production

- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Rate limiting configured
- [ ] Input validation in place
- [ ] Dependencies updated
- [ ] Security headers configured
- [ ] Logging and monitoring enabled
- [ ] Secrets stored securely (not in code)
- [ ] Regular security audits scheduled
- [ ] Incident response plan in place

## Known Security Considerations

### TensorFlow Security
- Keep TensorFlow updated to latest stable version
- Be aware of potential model poisoning attacks
- Validate model files from untrusted sources

### FastAPI Security
- Configure CORS appropriately for your use case
- Implement proper authentication
- Use HTTPS in production
- Enable rate limiting

## Security Tools

We use the following tools to maintain security:

- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability checker
- **Pre-commit hooks**: Automated security checks
- **GitHub Security Alerts**: Dependency vulnerability notifications
- **CodeQL**: Code security analysis (via GitHub Actions)

## Disclosure Policy

When we receive a security report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release new versions as soon as possible

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.
