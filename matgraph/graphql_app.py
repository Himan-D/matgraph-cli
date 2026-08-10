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
    true_form_energy: typing.Optional[float]
    predicted_form_energy: float
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
        model: typing.Optional[str] = "cgcnn",
        limit: typing.Optional[int] = None
    ) -> typing.List[MaterialPrediction]:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise Exception("MP_API_KEY environment variable is missing.")
            
        from matgraph.settings import settings
        eff_limit = limit if limit is not None else settings.graphql_default_limit
        eff_limit = min(eff_limit, settings.graphql_max_limit)
        raw_results = await asyncio.to_thread(
            run_pipeline, formula, api_key, min_gap, max_gap, crystal_system, model
        )
        
        graphql_results = []
        for r in raw_results[:eff_limit]:
            feats = MaterialFeatures(
                num_elements=r["features"]["num_elements"],
                mean_atomic_mass=r["features"]["mean_atomic_mass"],
                volume=r["features"]["volume"],
                density=r["features"]["density"]
            )
            metrics = ModelMetrics(
                model_name=f"PyTorch-{r['model_used']}-v1",
                confidence_score=0.94 if r['model_used'] == "CGCNN" else 0.92
            )
            graphql_results.append(
                MaterialPrediction(
                    material_id=strawberry.ID(str(r["material_id"])),
                    formula=r["formula"],
                    crystal_system=r["crystal_system"],
                    true_band_gap=r["true_band_gap"],
                    predicted_band_gap=r["predicted_band_gap"],
                    true_form_energy=r["true_form_energy"],
                    predicted_form_energy=r["predicted_form_energy"],
                    features=feats,
                    metrics=metrics
                )
            )
        return graphql_results
        
    @strawberry.field(description="Fetch Phonon Density of States (DOS)")
    async def phonon_dos(
        self,
        formula: str,
        method: typing.Optional[str] = "dfpt"
    ) -> "PhononDOS":
        from matgraph.core import fetch_phonon_dos
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise Exception("MP_API_KEY environment variable is missing.")
            
        raw_result = await asyncio.to_thread(
            fetch_phonon_dos, formula, api_key, method
        )
        
        return PhononDOS(
            material_id=strawberry.ID(raw_result["material_id"]),
            formula=raw_result["formula"],
            phonon_method=raw_result["phonon_method"],
            frequencies=raw_result["frequencies"],
            densities=raw_result["densities"]
        )

@strawberry.type
class PhononDOS:
    material_id: strawberry.ID
    formula: str
    phonon_method: str
    frequencies: typing.List[float]
    densities: typing.List[float]

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from matgraph.auth import is_valid_key

API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key and is_valid_key(api_key):
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key. Pass 'x-api-key' in headers.",
    )

app = FastAPI(
    title="MatGraph GraphQL Engine", 
    description="Modern, Async GraphQL API with advanced filtering"
)
app.include_router(graphql_app, prefix="/graphql", dependencies=[Depends(get_api_key)])

# REST fallback for researchers who prefer curl | jq
from pydantic import BaseModel
from matgraph.config import get_api_key as _cfg_get

class PredictRESTRequest(BaseModel):
    formula: str
    min_gap: typing.Optional[float] = None
    max_gap: typing.Optional[float] = None
    crystal_system: typing.Optional[str] = None
    model: str = "m3gnet"
    limit: typing.Optional[int] = None

@app.post("/v1/predict", dependencies=[Depends(get_api_key)])
async def rest_predict(req: PredictRESTRequest):
    import asyncio, os
    from matgraph.core import run_pipeline
    api_key = os.getenv("MP_API_KEY") or _cfg_get() or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="MP_API_KEY not configured on server")
    from matgraph.settings import settings
    eff_limit = req.limit if req.limit is not None else settings.graphql_default_limit
    eff_limit = min(eff_limit, settings.graphql_max_limit)
    results = await asyncio.to_thread(run_pipeline, req.formula, api_key, req.min_gap, req.max_gap, req.crystal_system, req.model)
    # strip non-serializable structure
    clean = [{k: v for k, v in r.items() if k != "structure"} for r in results[:eff_limit]]
    return {"count": len(clean), "results": clean}

@app.get("/health")
async def health():
    return {"status": "ok"}
