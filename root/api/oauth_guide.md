# OAuth2 Authentication Guide

## Overview
This guide explains how to authenticate API requests using OAuth2.

## Getting an Access Token

Make a POST request to the token endpoint:

```bash
curl -X POST https://api.example.com/auth/token \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"
```

## Using the Access Token

Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  https://api.example.com/data
```

## Token Expiration
Access tokens expire after 1 hour. Refresh tokens can be used to obtain new access tokens.
