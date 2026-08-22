from __future__ import annotations

import asyncio
import json
import operator
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, TypedDict, Annotated

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from psycopg import OperationalError as PsycopgOperationalError
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from mcp_client import client as mcp_client, tavily_mcp_search, weather_mcp_search, forecast_mcp_search

try:
    import airportsdata
    _IATA_AIRPORTS = airportsdata.load("IATA")
except Exception:
    _IATA_AIRPORTS = {}


LLM_TIMEOUT_SECONDS = 25
MCP_TIMEOUT_SECONDS = 15
# The weather MCP server runs on a separate free-tier Render service, which
# spins down after ~15 min idle and can take 30-60s to cold-start on the
# next request. The shared 15s MCP_TIMEOUT_SECONDS isn't enough for that --
# give weather its own longer budget instead of tightening the others.
WEATHER_TIMEOUT_SECONDS = 50

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    if "keepalives=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = (
            f"{database_url}{separator}"
            "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=3"
        )
    return database_url


llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)


async def _llm_json(system_prompt: str, user_prompt: str, timeout: float = LLM_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = await asyncio.wait_for(
        llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
        timeout=timeout,
    )
    text = str(response.content)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(text[start : end + 1])


async def _llm_text(system_prompt: str, user_prompt: str, timeout: float = LLM_TIMEOUT_SECONDS) -> str:
    response = await asyncio.wait_for(
        llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]),
        timeout=timeout,
    )
    return str(response.content)


def resolve_iata(city_or_airport: str) -> str | None:
    if not city_or_airport:
        return None

    query = city_or_airport.strip().upper()
    if len(query) == 3 and query in _IATA_AIRPORTS:
        return query

    query_lower = city_or_airport.strip().lower()
    matches = [
        code
        for code, info in _IATA_AIRPORTS.items()
        if info.get("city", "").lower() == query_lower
    ]
    if not matches:
        matches = [
            code
            for code, info in _IATA_AIRPORTS.items()
            if query_lower in info.get("city", "").lower() or query_lower in info.get("name", "").lower()
        ]
    return matches[0] if matches else None


class TripConstraints(TypedDict, total=False):
    origin: str
    destination: str
    departure_date: str
    return_date: str
    travelers: int
    cabin_class: str
    flight_preference: str
    nonstop: bool
    budget: str
    hotel_budget: str
    hotel_preference: str
    amenities: list[str]
    travel_style: str
    special_preferences: list[str]


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: TripConstraints
    supervisor_reasoning: str

    flight_results: str
    flight_matches: list[dict[str, Any]]
    flight_ranking: dict[str, Any]
    flight_booking_links: dict[str, Any]
    flight_limitations: str

    hotel_results: str
    hotel_candidates: list[dict[str, Any]]
    hotel_ranking: dict[str, Any]
    hotel_booking_links: dict[str, Any]

    weather_results: str
    budget_results: str
    itinerary: str

    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int
    agent_errors: dict[str, str]


KNOWN_AGENTS = {"flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"}
AGENT_ORDER = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"]


def _empty_constraints() -> TripConstraints:
    return {
        "origin": "",
        "destination": "",
        "departure_date": "",
        "return_date": "",
        "travelers": 1,
        "cabin_class": "",
        "flight_preference": "",
        "nonstop": False,
        "budget": "",
        "hotel_budget": "",
        "hotel_preference": "",
        "amenities": [],
        "travel_style": "",
        "special_preferences": [],
    }


SUPERVISOR_PROMPT = """
You are the routing brain for a multi-agent travel-planning system. Do two
things with the user's request below, and return ONLY strict JSON:

1. GUARDRAIL: decide if this is a legitimate travel-planning/travel-info
   request (destinations, flights, hotels, weather, budget, visas,
   itineraries, transport, food, packing, etc). Block only requests that are
   clearly unrelated to travel or that ask for harmful/illegal instructions.
   Do not block a valid travel request just because some details are missing.

2. If allowed, extract trip constraints and pick which specialist agents are
   needed:
   - flight_agent: needed if flights/airports/routes are relevant
   - hotel_agent: needed if accommodation is relevant
   - weather_agent: needed if weather/climate/packing is relevant
   - budget_agent: needed if cost/affordability is relevant
   - itinerary_agent: ALWAYS include

Return strict JSON only, no prose, using exactly this schema:
{{
  "allowed": true,
  "reason": "",
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "origin": "",
    "destination": "",
    "departure_date": "",
    "return_date": "",
    "travelers": 1,
    "cabin_class": "",
    "flight_preference": "",
    "nonstop": false,
    "budget": "",
    "hotel_budget": "",
    "hotel_preference": "",
    "amenities": [],
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

Dates should be normalized to YYYY-MM-DD when the user gives enough
information to do so; otherwise leave the field as an empty string — never
invent a date.

User request:
{query}
"""


async def supervisor_agent(state: TravelState) -> dict[str, Any]:
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    try:
        parsed = await _llm_json(
            "You are the routing and guardrail layer for a travel-planning app. "
            "Return strict JSON only.",
            SUPERVISOR_PROMPT.format(query=query),
        )
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        constraints = _empty_constraints()
        return {
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": AGENT_ORDER.copy(),
            "trip_constraints": constraints,
            "supervisor_reasoning": (
                "Supervisor parsing failed; the full agent set was selected as a "
                "safe fallback."
            ),
            "messages": [AIMessage(content="Supervisor fallback engaged.")],
            "llm_calls": llm_calls,
        }

    allowed = bool(parsed.get("allowed", True))
    reason = str(parsed.get("reason", "")).strip()

    if not allowed:
        final_reason = reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": final_reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": final_reason,
            "final_response": final_reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {final_reason}")],
            "llm_calls": llm_calls,
        }

    requested_agents = parsed.get("selected_agents", [])
    selected_agents = [a for a in AGENT_ORDER if a in requested_agents and a in KNOWN_AGENTS]
    if "itinerary_agent" not in selected_agents:
        selected_agents.append("itinerary_agent")

    constraints = _empty_constraints()
    raw_constraints = parsed.get("trip_constraints", {})
    if isinstance(raw_constraints, dict):
        constraints.update(raw_constraints)

    return {
        "guardrail_allowed": True,
        "guardrail_reason": reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": str(parsed.get("reasoning", "")).strip(),
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: TravelState) -> dict[str, Any]:
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {"final_response": reason, "messages": [AIMessage(content=reason)]}


async def _discover_aviation_tools() -> dict[str, Any]:
    tools = await asyncio.wait_for(
        mcp_client.get_tools(server_name="aviationstack"), timeout=MCP_TIMEOUT_SECONDS
    )
    return {t.name: t for t in tools}


def _rank_flight_matches(matches: list[dict[str, Any]], constraints: TripConstraints) -> dict[str, Any]:
    if not matches:
        return {"note": "No matching flight records were returned for this route."}

    preferred_airline = (constraints.get("flight_preference") or "").strip().lower()

    def has_time(m: dict[str, Any]) -> bool:
        return bool(m.get("departure_time") and m.get("arrival_time"))

    def duration_minutes(m: dict[str, Any]) -> float | None:
        try:
            dep = datetime.fromisoformat(m["departure_time"])
            arr = datetime.fromisoformat(m["arrival_time"])
            return (arr - dep).total_seconds() / 60.0
        except Exception:
            return None

    ranking: dict[str, Any] = {}

    airline_matched = [
        m for m in matches
        if preferred_airline and preferred_airline in (m.get("airline") or "").lower()
    ]
    if airline_matched:
        ranking["matches_airline_preference"] = airline_matched[:3]

    timed = [m for m in matches if has_time(m)]
    timed_with_duration = [(m, duration_minutes(m)) for m in timed]
    timed_with_duration = [(m, d) for m, d in timed_with_duration if d is not None]
    if timed_with_duration:
        timed_with_duration.sort(key=lambda pair: pair[1])
        ranking["shortest_scheduled_duration"] = [
            {**m, "scheduled_duration_minutes": round(d)} for m, d in timed_with_duration[:3]
        ]

    ranking["all_matches"] = matches[:10]
    ranking["total_matches_found"] = len(matches)
    return ranking


async def flight_agent(state: TravelState) -> dict[str, Any]:
    constraints = state.get("trip_constraints", {}) or {}
    origin_name = constraints.get("origin") or ""
    destination_name = constraints.get("destination") or ""
    departure_date = constraints.get("departure_date") or ""

    origin_iata = resolve_iata(origin_name)
    destination_iata = resolve_iata(destination_name)

    if not origin_iata or not destination_iata:
        missing = []
        if not origin_iata:
            missing.append(f"origin ('{origin_name or 'not provided'}')")
        if not destination_iata:
            missing.append(f"destination ('{destination_name or 'not provided'}')")
        message = (
            "Couldn't resolve an IATA airport code for: " + ", ".join(missing) + ". "
            "Ask the user for a specific city or airport name and try again."
        )
        return {
            "flight_results": message,
            "flight_matches": [],
            "flight_ranking": {},
            "flight_booking_links": {},
            "flight_limitations": message,
            "messages": [AIMessage(content="Flight agent could not resolve airport codes.")],
        }

    matches: list[dict[str, Any]] = []
    tools_found: list[str] = []
    tools_missing: list[str] = []
    error_note = ""

    try:
        available_tools = await _discover_aviation_tools()

        if "list_routes" in available_tools:
            tools_found.append("list_routes")
            routes_result = await asyncio.wait_for(
                available_tools["list_routes"].ainvoke(
                    {"dep_iata": origin_iata, "arr_iata": destination_iata, "limit": 10}
                ),
                timeout=MCP_TIMEOUT_SECONDS,
            )
            for record in _coerce_records(routes_result):
                matches.append(
                    {
                        "source": "list_routes",
                        "airline": record.get("airline_name") or record.get("airline") or "",
                        "departure_airport": origin_iata,
                        "arrival_airport": destination_iata,
                        "flight_number": record.get("flight_number", ""),
                        "departure_time": record.get("departure_time", ""),
                        "arrival_time": record.get("arrival_time", ""),
                    }
                )
        else:
            tools_missing.append("list_routes")

        if "historical_flights_by_date" in available_tools and departure_date:
            tools_found.append("historical_flights_by_date")
            hist_result = await asyncio.wait_for(
                available_tools["historical_flights_by_date"].ainvoke(
                    {
                        "flight_date": departure_date,
                        "number_of_flights": 10,
                        "dep_iata": origin_iata,
                        "arr_iata": destination_iata,
                    }
                ),
                timeout=MCP_TIMEOUT_SECONDS,
            )
            for record in _coerce_records(hist_result):
                matches.append(
                    {
                        "source": "historical_flights_by_date",
                        "airline": record.get("airline_name") or record.get("airline") or "",
                        "departure_airport": origin_iata,
                        "arrival_airport": destination_iata,
                        "flight_number": record.get("flight_number", ""),
                        "departure_time": record.get("departure_scheduled") or record.get("departure_time", ""),
                        "arrival_time": record.get("arrival_scheduled") or record.get("arrival_time", ""),
                    }
                )
        elif "historical_flights_by_date" not in available_tools:
            tools_missing.append("historical_flights_by_date")

    except asyncio.TimeoutError:
        error_note = "The aviation MCP server did not respond in time."
    except Exception as exc:
        error_note = f"Aviation MCP error: {type(exc).__name__}: {exc}"

    limitation_parts = [
        "AviationStack (via this MCP integration) does not provide fare pricing, "
        "seat availability, or booking capability — only schedule/route reference "
        "data. Any prices shown to the user must come from the booking link, not "
        "from this agent."
    ]
    if tools_missing:
        limitation_parts.append(
            f"The following expected tools were not found on the connected server: "
            f"{', '.join(tools_missing)}."
        )
    if error_note:
        limitation_parts.append(error_note)
    flight_limitations = " ".join(limitation_parts)

    ranking = _rank_flight_matches(matches, constraints)
    booking_links = _build_flight_booking_links(
        origin_iata, destination_iata, origin_name, destination_name, departure_date
    )

    if matches:
        summary = (
            f"Found {len(matches)} route/schedule record(s) between {origin_iata} and "
            f"{destination_iata} using: {', '.join(tools_found) or 'no live tools'}. "
            f"{flight_limitations}"
        )
    else:
        summary = (
            f"No route/schedule records were returned for {origin_iata} -> {destination_iata}. "
            f"{flight_limitations}"
        )

    return {
        "flight_results": summary,
        "flight_matches": matches,
        "flight_ranking": ranking,
        "flight_booking_links": booking_links,
        "flight_limitations": flight_limitations,
        "messages": [AIMessage(content="Flight route search completed.")],
    }


def _coerce_records(result: Any) -> list[dict[str, Any]]:
    """MCP tool results can come back as a list, a dict with a 'data' key, or
    a JSON string depending on server version — normalize without inventing
    fields that aren't there."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return []
    if isinstance(result, dict):
        result = result.get("data", result.get("results", []))
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


def _build_flight_booking_links(
    origin_iata: str, destination_iata: str, origin_name: str, destination_name: str, departure_date: str
) -> dict[str, Any]:
    query_bits = [f"Flights from {origin_name or origin_iata} to {destination_name or destination_iata}"]
    if departure_date:
        query_bits.append(f"on {departure_date}")
    query = " ".join(query_bits)
    url = "https://www.google.com/travel/flights?q=" + query.replace(" ", "%20")
    return {
        "type": "search_link",
        "provider": "Google Flights",
        "url": url,
        "disclaimer": "This opens a live search — it does not guarantee the price or availability shown.",
    }


HOTEL_EXTRACTION_PROMPT = """
Below is raw web search text about hotels. Extract ONLY hotels that are
explicitly named in the text. Do not invent hotels, prices, or ratings.
If a field isn't mentioned for a given hotel, use null for that field —
never guess a plausible-sounding value.

Return strict JSON only:
{{
  "hotels": [
    {{"name": "", "price_mentioned": null, "rating_mentioned": null, "location_note": null, "notable_feature": null}}
  ]
}}

Search text:
{text}
"""


def _rank_hotel_candidates(hotels: list[dict[str, Any]]) -> dict[str, Any]:
    if not hotels:
        return {"note": "No hotels were confidently extracted from the search results."}

    ranking: dict[str, Any] = {"all_candidates": hotels}

    def parse_price(h: dict[str, Any]) -> float | None:
        raw = h.get("price_mentioned")
        if not raw:
            return None
        digits = "".join(ch for ch in str(raw) if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else None
        except ValueError:
            return None

    def parse_rating(h: dict[str, Any]) -> float | None:
        raw = h.get("rating_mentioned")
        if not raw:
            return None
        digits = "".join(ch for ch in str(raw) if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else None
        except ValueError:
            return None

    priced = [(h, parse_price(h)) for h in hotels]
    priced = [(h, p) for h, p in priced if p is not None]
    if priced:
        priced.sort(key=lambda pair: pair[1])
        ranking["best_budget"] = priced[0][0]

    rated = [(h, parse_rating(h)) for h in hotels]
    rated = [(h, r) for h, r in rated if r is not None]
    if rated:
        rated.sort(key=lambda pair: pair[1], reverse=True)
        ranking["best_rated"] = rated[0][0]

    if not priced and not rated:
        ranking["note"] = (
            "The search results didn't include explicit prices or ratings for these "
            "hotels — showing them unranked. Use the booking link for live pricing."
        )

    return ranking


def _build_hotel_booking_links(destination: str, checkin: str, checkout: str, travelers: int) -> dict[str, Any]:
    params = [f"ss={(destination or '').replace(' ', '+')}"]
    if checkin:
        params.append(f"checkin={checkin}")
    if checkout:
        params.append(f"checkout={checkout}")
    params.append(f"group_adults={max(travelers or 1, 1)}")
    url = "https://www.booking.com/searchresults.html?" + "&".join(params)
    return {
        "type": "search_link",
        "provider": "Booking.com",
        "url": url,
        "disclaimer": "This opens a live search — it does not guarantee the price or availability shown.",
    }


async def hotel_agent(state: TravelState) -> dict[str, Any]:
    constraints = state.get("trip_constraints", {}) or {}
    destination = constraints.get("destination") or ""

    query_parts = [f"best hotels in {destination}" if destination else state["user_query"]]
    if constraints.get("hotel_budget"):
        query_parts.append(f"under {constraints['hotel_budget']}")
    if constraints.get("hotel_preference"):
        query_parts.append(constraints["hotel_preference"])
    if constraints.get("amenities"):
        query_parts.append("with " + ", ".join(constraints["amenities"]))
    search_query = " ".join(str(p) for p in query_parts if p)

    hotel_candidates: list[dict[str, Any]] = []
    llm_calls_used = 0

    try:
        raw_results = await asyncio.wait_for(tavily_mcp_search(search_query), timeout=MCP_TIMEOUT_SECONDS)
        extraction = await _llm_json(
            "You extract only explicitly-stated facts from search text. Return strict JSON only.",
            HOTEL_EXTRACTION_PROMPT.format(text=str(raw_results)[:6000]),
        )
        llm_calls_used += 1
        hotel_candidates = [h for h in extraction.get("hotels", []) if isinstance(h, dict) and h.get("name")]
        hotel_results_text = f"Found {len(hotel_candidates)} hotel(s) mentioned in live search results for {destination}."
    except asyncio.TimeoutError:
        hotel_results_text = "Live hotel search timed out. No hotel data is available for this request."
    except Exception as exc:
        print(f"HOTEL AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        hotel_results_text = (
            "Live hotel search is temporarily unavailable, and no hotel data could be "
            "retrieved. General, non-live advice is not provided here to avoid implying "
            "these are real options."
        )

    ranking = _rank_hotel_candidates(hotel_candidates)
    booking_links = _build_hotel_booking_links(
        destination, constraints.get("departure_date", ""), constraints.get("return_date", ""),
        constraints.get("travelers", 1),
    )

    return {
        "hotel_results": hotel_results_text,
        "hotel_candidates": hotel_candidates,
        "hotel_ranking": ranking,
        "hotel_booking_links": booking_links,
        "messages": [AIMessage(content="Hotel search processed.")],
        "llm_calls": state.get("llm_calls", 0) + llm_calls_used,
    }


async def weather_agent(state: TravelState) -> dict[str, Any]:
    constraints = state.get("trip_constraints", {}) or {}
    city = constraints.get("destination") or ""

    if not city:
        message = "No destination was extracted from the request, so weather data was skipped."
        return {"weather_results": message, "messages": [AIMessage(content=message)]}

    try:
        weather_data, forecast_data = await asyncio.wait_for(
            asyncio.gather(weather_mcp_search(city), forecast_mcp_search(city)),
            timeout=WEATHER_TIMEOUT_SECONDS,
        )
        weather_results = f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}"
    except asyncio.TimeoutError:
        weather_results = f"Live weather lookup for {city} timed out. Advise the traveler to check a live forecast before departure."
    except Exception as exc:
        print(f"WEATHER AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        weather_results = (
            f"Live weather information for {city} is temporarily unavailable. "
            "Advise the traveler to verify the forecast before departure."
        )

    return {"weather_results": weather_results, "messages": [AIMessage(content="Weather information processed.")]}


async def data_agents_node(state: TravelState) -> dict[str, Any]:
    selected = set(state.get("selected_agents", []))
    tasks: dict[str, Any] = {}

    if "flight_agent" in selected:
        tasks["flight_agent"] = flight_agent(state)
    if "hotel_agent" in selected:
        tasks["hotel_agent"] = hotel_agent(state)
    if "weather_agent" in selected:
        tasks["weather_agent"] = weather_agent(state)

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    merged: dict[str, Any] = {}
    errors = dict(state.get("agent_errors", {}))
    llm_calls = state.get("llm_calls", 0)
    all_messages: list[AnyMessage] = []

    for name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"{name} raised: {type(result).__name__}: {result}", flush=True)
            errors[name] = f"{type(result).__name__}: {result}"
            continue
        for key, value in result.items():
            if key == "messages":
                all_messages.extend(value)
            elif key == "llm_calls":
                llm_calls = max(llm_calls, value)
            else:
                merged[key] = value

    merged["messages"] = all_messages
    merged["llm_calls"] = llm_calls
    merged["agent_errors"] = errors
    return merged


async def budget_agent(state: TravelState) -> dict[str, Any]:
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Info (schedule data only — no pricing available, see flight_limitations):
{state.get('flight_results', '')}

Hotel Info:
{state.get('hotel_results', '')}

Weather Info:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories (label clearly as rough estimates, since no live
   flight/hotel pricing is available from the connected tools)
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

Do not state or imply a specific flight or hotel price as fact.
"""
    try:
        text = await _llm_text("You are a practical travel budget analyst.", prompt)
    except asyncio.TimeoutError:
        text = "Budget analysis timed out and was skipped for this run."

    return {
        "budget_results": text,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def itinerary_agent(state: TravelState) -> dict[str, Any]:
    prompt = f"""
Create a complete travel itinerary draft.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Info:
{state.get('flight_results', '')}
Flight limitations: {state.get('flight_limitations', '')}

Hotel Info:
{state.get('hotel_results', '')}

Weather Info:
{state.get('weather_results', '')}

Budget Info:
{state.get('budget_results', '')}

Make the itinerary practical and easy to follow. Do not state a flight or
hotel price as confirmed fact — point the user to the booking links for
live pricing. Create a clear draft ready for human review.
"""
    text = await _llm_text("You are an expert travel planner.", prompt)

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": text,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def human_approval_agent(state: TravelState) -> dict[str, Any]:
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "flight_booking_links": state.get("flight_booking_links", {}),
            "hotel_booking_links": state.get("hotel_booking_links", {}),
            "expected_response": {"approved": True, "feedback": "Optional revision feedback"},
        }
    )
    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()
    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


async def final_agent(state: TravelState) -> dict[str, Any]:
    if state.get("approved", False):
        review_instruction = "The user approved the draft. Preserve its decisions while polishing it."
    else:
        review_instruction = (
            "The user requested a revision. Apply this feedback carefully:\n"
            f"{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}"
        )

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}
Flight limitations: {state.get('flight_limitations', '')}
Flight search link: {state.get('flight_booking_links', {})}

Hotels:
{state.get('hotel_results', '')}
Hotel search link: {state.get('hotel_booking_links', {})}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer using these sections:
1. Trip Summary
2. Flight Information (state clearly that live pricing isn't available and
   link to the search link for booking)
3. Hotel Suggestions (same — link to the search link)
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget (label as an estimate)
7. Final Recommendations

Never state a flight/hotel price, rating, or availability as a confirmed
fact unless it was explicitly present in the Flights/Hotels data above.
Incorporate the human feedback when a revision was requested.
"""
    response = await asyncio.wait_for(
        llm.ainvoke(
            [SystemMessage(content="You are a professional AI travel booking assistant."), HumanMessage(content=final_prompt)]
        ),
        timeout=LLM_TIMEOUT_SECONDS,
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_from_supervisor(state: TravelState) -> str:
    return "guardrail_blocked" if not state.get("guardrail_allowed", True) else "data_agents"


def route_after_data_agents(state: TravelState) -> str:
    selected = set(state.get("selected_agents", []))
    return "budget_agent" if "budget_agent" in selected else "itinerary_agent"


def build_travel_graph(checkpointer: AsyncPostgresSaver):
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("guardrail_blocked", guardrail_blocked_agent)
    graph.add_node("data_agents", data_agents_node)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_agent", final_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_from_supervisor, {"guardrail_blocked": "guardrail_blocked", "data_agents": "data_agents"}
    )
    graph.add_conditional_edges(
        "data_agents", route_after_data_agents, {"budget_agent": "budget_agent", "itinerary_agent": "itinerary_agent"}
    )
    graph.add_edge("budget_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_agent")
    graph.add_edge("final_agent", END)
    graph.add_edge("guardrail_blocked", END)

    return graph.compile(checkpointer=checkpointer)


class TravelBackend:

    def __init__(self) -> None:
        self.pool: AsyncConnectionPool | None = None
        self.checkpointer: AsyncPostgresSaver | None = None
        self.graph = None

    async def init(self, min_size: int = 2, max_size: int = 10) -> None:
        database_url = get_database_url()
        self.pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            max_idle=180,
            max_lifetime=1800,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        await self.pool.open(wait=True, timeout=30)

        self.checkpointer = AsyncPostgresSaver(self.pool)
        await self.checkpointer.setup()

        self.graph = build_travel_graph(self.checkpointer)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    async def _ainvoke_with_retry(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self.graph.ainvoke(payload, config=config)
        except PsycopgOperationalError as exc:
            print(f"DB connection was stale, retrying once: {exc}")
            return await self.graph.ainvoke(payload, config=config)

    async def run_travel_agent(self, user_input: str, thread_id: str | None = None) -> dict[str, Any]:
        if not thread_id:
            thread_id = f"user_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}

        result = await self._ainvoke_with_retry(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "trip_constraints": _empty_constraints(),
                "supervisor_reasoning": "",
                "flight_results": "",
                "flight_matches": [],
                "flight_ranking": {},
                "flight_booking_links": {},
                "flight_limitations": "",
                "hotel_results": "",
                "hotel_candidates": [],
                "hotel_ranking": {},
                "hotel_booking_links": {},
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "approval_request": "",
                "approved": False,
                "human_feedback": "",
                "final_response": "",
                "llm_calls": 0,
                "agent_errors": {},
            },
            config=config,
        )
        return _serialize_result(result, thread_id)

    async def resume_travel_agent(self, thread_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
        if not thread_id:
            raise ValueError("thread_id is required to resume a travel plan.")
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._ainvoke_with_retry(
            Command(resume={"approved": approved, "feedback": feedback.strip()}),
            config,
        )
        return _serialize_result(result, thread_id)


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get("itinerary", "")

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "") if interrupt_payload else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "flight_ranking": result.get("flight_ranking", {}),
        "flight_booking_links": result.get("flight_booking_links", {}),
        "flight_limitations": result.get("flight_limitations", ""),
        "hotel_results": result.get("hotel_results", ""),
        "hotel_ranking": result.get("hotel_ranking", {}),
        "hotel_booking_links": result.get("hotel_booking_links", {}),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": interrupt_payload.get("draft_itinerary", "") if interrupt_payload else result.get("itinerary", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "agent_errors": result.get("agent_errors", {}),
        "llm_calls": result.get("llm_calls", 0),
    }


_backend: TravelBackend | None = None


async def init_travel_backend() -> TravelBackend:
    global _backend
    if _backend is None:
        _backend = TravelBackend()
        await _backend.init()
    return _backend


async def close_travel_backend() -> None:
    global _backend
    if _backend is not None:
        await _backend.close()
        _backend = None


def get_travel_backend() -> TravelBackend:
    if _backend is None:
        raise RuntimeError(
            "Travel backend is not initialized. Call `await init_travel_backend()` "
            "during FastAPI startup (lifespan) before handling requests."
        )
    return _backend


async def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict[str, Any]:
    return await get_travel_backend().run_travel_agent(user_input, thread_id)


async def resume_travel_agent(thread_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
    return await get_travel_backend().resume_travel_agent(thread_id, approved, feedback)