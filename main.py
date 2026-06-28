from fastapi import FastAPI
from pydantic import BaseModel
import sys
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List
from src.monitor import log_prediction
sys.path.insert(0, '/app')
from src.predict import predict_priority

import src.monitor as mon
print("LOG PATH:", mon.LOG_PATH)

app = FastAPI()



class TicketInput(BaseModel):
    impact: int
    urgency: int
    reassignment_count: int
    reopen_count: int
    sys_mod_count: int
    notify: int
    contact_type: str
    category: str
    subcategory: str
    opened_at: str
    location: str = "Unknown"
    assignment_group: str = "Unassigned"

class BatchInput(BaseModel):
    tickets: List[TicketInput]

@app.get("/")
def root():
    return {"message": "IT Incident Priority Predictor API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(ticket: TicketInput):
    label, confidence, probability, _ = predict_priority(
        impact=ticket.impact,
        urgency=ticket.urgency,
        reassignment_count=ticket.reassignment_count,
        reopen_count=ticket.reopen_count,
        contact_type=ticket.contact_type,
        category=ticket.category,
        subcategory=ticket.subcategory,
        opened_at=ticket.opened_at,
        sys_mod_count=ticket.sys_mod_count,
        notify=ticket.notify,
        location=ticket.location,
        assignment_group=ticket.assignment_group,
    )
    try:
        log_prediction(ticket.model_dump(), str(label), float(probability))
    except Exception as e:
        print(f"Logging error: {e}")
    return {
        "label": str(label),
        "confidence": float(confidence),
        "probability": float(probability),
    }

def run_prediction(ticket: TicketInput):
    return predict_priority(
        impact=ticket.impact,
        urgency=ticket.urgency,
        reassignment_count=ticket.reassignment_count,
        reopen_count=ticket.reopen_count,
        contact_type=ticket.contact_type,
        category=ticket.category,
        subcategory=ticket.subcategory,
        opened_at=ticket.opened_at,
        sys_mod_count=ticket.sys_mod_count,
        notify=ticket.notify,
        location=ticket.location,
        assignment_group=ticket.assignment_group,
    )

@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    loop = asyncio.get_event_loop()
    
    with ProcessPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, run_prediction, ticket)
            for ticket in batch.tickets
        ]
        results = await asyncio.gather(*futures)
    
    return [
        {
            "label": str(r[0]),
            "confidence": float(r[1]),
            "probability": float(r[2])
        }
        for r in results
    ]