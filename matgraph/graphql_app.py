import typing
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import os
import asyncio
from matgraph.core import run_pipeline

@strawberry.type
class MaterialFeatures:
    num_elements: int
    mean_atomic_mass: float
    volume: float
    density: float

@strawberry.type
class ModelMetrics:
    model_name: str
    confidence_score: float

@strawberry.type
class MaterialPrediction:
    material_id: strawberry.ID
    formula: str
    crystal_system: str
    true_band_gap: typing.Optional[float]
    predicted_band_gap: float
    features: MaterialFeatures
    metrics: ModelMetrics

@strawberry.type
class Query:
    @strawberry.field(description="Run ML pipeline with advanced material filters (band gap, crystal system)")
    async def predict_material(
        self, 
        formula: str,
        min_gap: typing.Optional[float] = None,
        max_gap: typing.Optional[float] = None,
        crystal_system: typing.Optional[str] = None,
        limit: typing.Optional[int] = 10
    ) -> typing.List[MaterialPrediction]:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise Exception("MP_API_KEY environment variable is missing.")
            
        raw_results = await asyncio.to_thread(
            run_pipeline, formula, api_key, min_gap, max_gap, crystal_system
        )
        
        graphql_results = []
        for r in raw_results[:limit]:
            feats = MaterialFeatures(
                num_elements=r["features"]["num_elements"],
                mean_atomic_mass=r["features"]["mean_atomic_mass"],
                volume=r["features"]["volume"],
                density=r["features"]["density"]
            )
            metrics = ModelMetrics(
                model_name="MatGraph-Dummy-RF-v2",
                confidence_score=0.91
            )
            graphql_results.append(
                MaterialPrediction(
                    material_id=strawberry.ID(str(r["material_id"])),
                    formula=r["formula"],
                    crystal_system=r["crystal_system"],
                    true_band_gap=r["true_band_gap"],
                    predicted_band_gap=r["predicted_band_gap"],
                    features=feats,
                    metrics=metrics
                )
            )
        return graphql_results

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI(
    title="🚀 MatGraph GraphQL Engine", 
    description="Modern, Async GraphQL API with advanced filtering"
)
app.include_router(graphql_app, prefix="/graphql")
