# API Examples

## Reset

```http
POST /reset
Content-Type: application/json
```

## Step

```http
POST /step
Content-Type: application/json

{
  "orders": [10.0, 10.0, 10.0]
}
```

## State

```http
GET /state
```

The exact request and response schemas are defined by the running OpenEnv server. Use the repository's validation and tests as the source of truth when extending the API.
