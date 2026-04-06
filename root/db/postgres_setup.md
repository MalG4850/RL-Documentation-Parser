# PostgreSQL Setup Guide

## Default Connection Parameters

| Parameter | Default Value |
|-----------|---------------|
| Host | localhost |
| Port | 5432 |
| Database | postgres |
| User | postgres |

## Connection String Format
```
postgresql://user:password@host:port/database
```

## Basic Connection Test
```bash
psql -h localhost -p 5432 -U postgres
```
