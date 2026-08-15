# GraphQL API Reference

MatGraph includes an async GraphQL API built on FastAPI and Strawberry for integrating predictions into web applications.

## Starting the Server

```bash
uvicorn matgraph.graphql_app:app --reload
```

The server starts at `http://localhost:8000`. The interactive GraphiQL playground is available at `http://localhost:8000/graphql`.

## Authentication

All requests require an `x-api-key` header.

### Generate a Key

```bash
matgraph auth generate --user "my-app"
# Output: mg_S8jvhzo58p6XQE_fS0Oru1WqG2AJR6x5
```

### Master Key (Alternative)

Set a master key via environment variable:

```bash
export MATGRAPH_API_KEY="my_master_key"
```

---

## Schema

### Query: predictMaterial

```graphql
query {
  predictMaterial(
    formula: String!
    minGap: Float
    maxGap: Float
    crystalSystem: String
    model: String = "cgcnn"
    limit: Int = 10
  ): [MaterialPrediction!]!
}
```

### Types

```graphql
type MaterialPrediction {
  materialId: ID!
  formula: String!
  crystalSystem: String!
  trueBandGap: Float
  predictedBandGap: Float!
  trueFormEnergy: Float
  predictedFormEnergy: Float!
  features: MaterialFeatures!
  metrics: ModelMetrics!
}

type MaterialFeatures {
  numElements: Int!
  meanAtomicMass: Float!
  volume: Float!
  density: Float!
}

type ModelMetrics {
  modelName: String!
  confidenceScore: Float!
}
```

---

## Example Queries

### Basic Prediction

```graphql
query {
  predictMaterial(formula: "LiFePO4") {
    materialId
    formula
    predictedBandGap
    predictedFormEnergy
  }
}
```

### With Filters

```graphql
query {
  predictMaterial(formula: "NaCl", minGap: 1.0, limit: 3, model: "megnet") {
    materialId
    formula
    crystalSystem
    trueBandGap
    predictedBandGap
    features {
      density
      volume
    }
    metrics {
      modelName
      confidenceScore
    }
  }
}
```

### cURL Example

```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{
    "query": "query { predictMaterial(formula: \"LiFePO4\", model: \"megnet\") { formula predictedBandGap predictedFormEnergy } }"
  }'
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/graphql",
    json={"query": '{ predictMaterial(formula: "LiFePO4") { formula predictedBandGap } }'},
    headers={"x-api-key": "YOUR_KEY"}
)
print(response.json())
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/graphql", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-api-key": "YOUR_KEY",
  },
  body: JSON.stringify({
    query: `{ predictMaterial(formula: "LiFePO4") { formula predictedBandGap } }`,
  }),
});
const data = await response.json();
console.log(data);
```
