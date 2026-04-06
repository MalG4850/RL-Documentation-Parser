# SSL Configuration Guide

## Overview
This document describes how to configure SSL certificates in the network module.

## Configuration Steps

1. Obtain your SSL certificate and private key files
2. Add the following parameters to your network configuration:

```yaml
ssl_enabled: true
cert_path: /path/to/certificate.crt
key_path: /path/to/private.key
```

3. Restart the network service for changes to take effect

## Certificate Requirements
- The certificate must be valid and signed by a trusted CA
- The private key must be in PEM format
