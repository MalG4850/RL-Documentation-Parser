# Database Query Optimization

## Performance Tuning Guide

### Using EXPLAIN ANALYZE

Run EXPLAIN ANALYZE before your query to see the execution plan:

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';
```

### Indexing Strategies

1. **Create indexes** on columns used in WHERE clauses:
```sql
CREATE INDEX idx_users_email ON users(email);
```

2. **Composite indexes** for multi-column queries:
```sql
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
```

### Connection Pooling

Use connection pooling to reduce overhead:

```python
from psycopg2.pool import ThreadedConnectionPool
pool = ThreadedConnectionPool(minconn=5, maxconn=20, 
                               host="localhost", database="mydb")
```

### Best Practices
- Avoid SELECT * - specify only needed columns
- Use LIMIT for large result sets
- Implement pagination for user-facing queries
- Cache frequently accessed data
