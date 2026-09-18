# =============================================================================
# Destination Planning Graph -- LangGraph Learning Project
# =============================================================================
#
# A user provides:
#   - Destination
#   - Trip duration
#   - Preferences
#   - Budget
#
# The graph runs 3 specialist planners IN PARALLEL:
#
#   1. plan_attractions
#   2. plan_food_and_culture
#   3. plan_trip_logistics
#
# Once all three specialists finish, select_itinerary_style decides whether
# the trip should be RELAXED or ACTIVITY-PACKED.
#
# The graph then conditionally routes to the appropriate itinerary generator.
#
#
# =============================================================================


import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


class TravelState(BaseModel):

    # User inputs
    destination: str = ""
    duration_days: int = 0
    preferences: str = ""
    budget: str = ""

    # Parallel specialist outputs
    attractions_plan: str = ""
    food_culture_plan: str = ""
    logistics_plan: str = ""

    # Decision node outputs
    itinerary_style: str = ""
    style_reason: str = ""

    # Final output
    final_itinerary: str = ""

    # Execution log
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)



def plan_attractions(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a travel sightseeing and experiences specialist.\n\n"

        f"Destination: {state.destination}\n"
        f"Trip duration: {state.duration_days} days\n"
        f"Traveler preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"Suggest attractions, sightseeing locations, and experiences "
        f"that match this traveler's interests and trip duration.\n\n"

        f"Focus on:\n"
        f"- Important attractions\n"
        f"- Experiences matching the traveler's preferences\n"
        f"- A reasonable mix of popular and distinctive experiences\n"
        f"- Activities appropriate for the stated budget\n\n"

        f"Do NOT claim live pricing, opening hours, schedules, "
        f"ticket availability, or reservation availability because "
        f"you do not have access to current data."
    )

    return {
        "attractions_plan": response.content,
        "messages": ["[plan_attractions] Done"]
    }


def plan_food_and_culture(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a local food and culture travel specialist.\n\n"

        f"Destination: {state.destination}\n"
        f"Trip duration: {state.duration_days} days\n"
        f"Traveler preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"Suggest suitable food and cultural experiences for this trip.\n\n"

        f"Consider:\n"
        f"- Local dishes worth trying\n"
        f"- Markets and food areas\n"
        f"- Cultural experiences\n"
        f"- Neighborhoods worth exploring\n"
        f"- Local traditions or experiences relevant to the destination\n"
        f"- Options appropriate for the traveler's budget\n\n"

        f"Do NOT claim live pricing, opening hours, schedules, "
        f"availability, or reservation availability because "
        f"you do not have access to current data."
    )

    return {
        "food_culture_plan": response.content,
        "messages": ["[plan_food_and_culture] Done"]
    }


def plan_trip_logistics(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a trip logistics and itinerary planning specialist.\n\n"

        f"Destination: {state.destination}\n"
        f"Trip duration: {state.duration_days} days\n"
        f"Traveler preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"Provide logistical recommendations for organizing this trip.\n\n"

        f"Consider:\n"
        f"- Geographic grouping of attractions\n"
        f"- Avoiding unnecessary backtracking\n"
        f"- General transportation strategy\n"
        f"- Daily pacing\n"
        f"- Travel time between areas\n"
        f"- Budget considerations\n"
        f"- Whether some days should be lighter than others\n\n"

        f"Do NOT claim live transportation prices, schedules, "
        f"opening hours, ticket availability, or reservation availability "
        f"because you do not have access to current data."
    )

    return {
        "logistics_plan": response.content,
        "messages": ["[plan_trip_logistics] Done"]
    }


def select_itinerary_style(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a travel itinerary decision system.\n\n"

        f"USER PROFILE\n"
        f"Destination: {state.destination}\n"
        f"Duration: {state.duration_days} days\n"
        f"Preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"Three travel specialists have produced recommendations.\n\n"

        f"----------------------------------------\n"
        f"ATTRACTIONS SPECIALIST\n"
        f"----------------------------------------\n"
        f"{state.attractions_plan}\n\n"

        f"----------------------------------------\n"
        f"FOOD AND CULTURE SPECIALIST\n"
        f"----------------------------------------\n"
        f"{state.food_culture_plan}\n\n"

        f"----------------------------------------\n"
        f"LOGISTICS SPECIALIST\n"
        f"----------------------------------------\n"
        f"{state.logistics_plan}\n\n"

        f"Decide which itinerary style best matches this traveler.\n\n"

        f"Choose ONLY one of these:\n\n"

        f"relaxed\n"
        f"- Fewer scheduled activities\n"
        f"- Comfortable daily pacing\n"
        f"- More free time\n"
        f"- Longer time at individual locations\n\n"

        f"packed\n"
        f"- More sightseeing and experiences each day\n"
        f"- Fuller schedules\n"
        f"- Suitable for travelers who want to maximize activities\n\n"

        f"Reply STRICTLY using this JSON format and nothing else:\n\n"

        f'{{'
        f'"itinerary_style": "relaxed" or "packed", '
        f'"reason": "one sentence explanation"'
        f'}}'
    )

    try:

        result = json.loads(response.content)

        style = result["itinerary_style"].lower()
        reason = result["reason"]

        # Validate the LLM response
        if style not in ("relaxed", "packed"):
            style = "relaxed"
            reason = (
                "The itinerary style could not be determined, "
                "so a relaxed itinerary was selected by default."
            )

    except (json.JSONDecodeError, KeyError):

        style = "relaxed"
        reason = (
            "The itinerary style decision could not be parsed, "
            "so a relaxed itinerary was selected by default."
        )

    return {
        "itinerary_style": style,
        "style_reason": reason,
        "messages": [
            f"[select_itinerary_style] style={style}"
        ]
    }


def route_itinerary(state: TravelState) -> str:

    if state.itinerary_style == "packed":
        return "packed"

    return "relaxed"


def relaxed_itinerary(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a friendly travel itinerary planner.\n\n"

        f"Create a RELAXED {state.duration_days}-day itinerary "
        f"for {state.destination}.\n\n"

        f"Traveler preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"The itinerary style decision was:\n"
        f"{state.style_reason}\n\n"

        f"Use the following specialist recommendations.\n\n"

        f"----------------------------------------\n"
        f"ATTRACTIONS\n"
        f"----------------------------------------\n"
        f"{state.attractions_plan}\n\n"

        f"----------------------------------------\n"
        f"FOOD AND CULTURE\n"
        f"----------------------------------------\n"
        f"{state.food_culture_plan}\n\n"

        f"----------------------------------------\n"
        f"LOGISTICS\n"
        f"----------------------------------------\n"
        f"{state.logistics_plan}\n\n"

        f"Create a day-by-day itinerary.\n\n"

        f"For each day include:\n"
        f"- Morning\n"
        f"- Afternoon\n"
        f"- Evening\n"
        f"- Food or cultural suggestions where appropriate\n\n"

        f"Keep the trip comfortable and relaxed.\n"
        f"Avoid overloading individual days.\n"
        f"Group nearby activities together where possible.\n"
        f"Leave some free time for spontaneous exploration.\n\n"

        f"Do NOT present estimated information as live information. "
        f"Do not claim current prices, opening hours, schedules, "
        f"availability, or reservations."
    )

    final_output = (
        f"RELAXED ITINERARY\n"
        f"{'=' * 55}\n"
        f"{response.content}"
    )

    return {
        "final_itinerary": final_output,
        "messages": ["[relaxed_itinerary] Generated"]
    }


def activity_packed_itinerary(state: TravelState) -> dict:

    response = llm.invoke(
        f"You are a friendly travel itinerary planner.\n\n"

        f"Create an ACTIVITY-PACKED {state.duration_days}-day itinerary "
        f"for {state.destination}.\n\n"

        f"Traveler preferences: {state.preferences}\n"
        f"Budget: {state.budget}\n\n"

        f"The itinerary style decision was:\n"
        f"{state.style_reason}\n\n"

        f"Use the following specialist recommendations.\n\n"

        f"----------------------------------------\n"
        f"ATTRACTIONS\n"
        f"----------------------------------------\n"
        f"{state.attractions_plan}\n\n"

        f"----------------------------------------\n"
        f"FOOD AND CULTURE\n"
        f"----------------------------------------\n"
        f"{state.food_culture_plan}\n\n"

        f"----------------------------------------\n"
        f"LOGISTICS\n"
        f"----------------------------------------\n"
        f"{state.logistics_plan}\n\n"

        f"Create a full day-by-day itinerary.\n\n"

        f"For each day include:\n"
        f"- Morning\n"
        f"- Afternoon\n"
        f"- Evening\n"
        f"- Food or cultural suggestions where appropriate\n\n"

        f"Fit in more sightseeing and experiences while still keeping "
        f"the itinerary geographically sensible.\n"

        f"Group nearby attractions together.\n"
        f"Avoid unnecessary backtracking.\n"
        f"Keep the schedule busy but realistic.\n\n"

        f"Do NOT present estimated information as live information. "
        f"Do not claim current prices, opening hours, schedules, "
        f"availability, or reservations."
    )

    final_output = (
        f"ACTIVITY-PACKED ITINERARY\n"
        f"{'=' * 55}\n"
        f"{response.content}"
    )

    return {
        "final_itinerary": final_output,
        "messages": ["[activity_packed_itinerary] Generated"]
    }


graph = StateGraph(TravelState)


# Add specialist nodes
graph.add_node("plan_attractions",plan_attractions)

graph.add_node("plan_food_and_culture",plan_food_and_culture)

graph.add_node("plan_trip_logistics",plan_trip_logistics)

# Add decision node
graph.add_node("select_itinerary_style",select_itinerary_style)

# Add final nodes
graph.add_node("relaxed_itinerary",relaxed_itinerary)

graph.add_node("activity_packed_itinerary",activity_packed_itinerary)

graph.add_edge(START,"plan_attractions")

graph.add_edge(START,"plan_food_and_culture")

graph.add_edge(START,"plan_trip_logistics")

graph.add_edge("plan_attractions","select_itinerary_style")

graph.add_edge("plan_food_and_culture","select_itinerary_style")

graph.add_edge("plan_trip_logistics","select_itinerary_style")

graph.add_conditional_edges("select_itinerary_style",
    route_itinerary,
    {
        "relaxed": "relaxed_itinerary",
        "packed": "activity_packed_itinerary",
    }
)

graph.add_edge("relaxed_itinerary",END)

graph.add_edge("activity_packed_itinerary",END)

app = graph.compile()


# =============================================================================
# RUN TRIP PLANNER
# =============================================================================

def run_trip_planner(
    destination: str,
    duration_days: int,
    preferences: str,
    budget: str
):

    print("\n" + "=" * 60)
    print(" AI DESTINATION PLANNER")
    print("=" * 60)

    print(f"  Destination : {destination}")
    print(f"  Duration    : {duration_days} days")
    print(f"  Preferences : {preferences}")
    print(f"  Budget      : {budget}")

    print("=" * 60)

    print("\n  Planning your trip...")

    result = app.invoke({
        "destination": destination,
        "duration_days": duration_days,
        "preferences": preferences,
        "budget": budget,
        "messages": [],
    })


    # -------------------------------------------------------------------------
    # Show decision
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  ITINERARY STYLE")
    print("=" * 60)

    print(
        f"\n  Selected style: "
        f"{result['itinerary_style'].upper()}"
    )

    print(
        f"  Reason: {result['style_reason']}"
    )


    # -------------------------------------------------------------------------
    # Show final itinerary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  YOUR PERSONALIZED ITINERARY")
    print("=" * 60)

    print(
        f"\n{result['final_itinerary']}"
    )


    # -------------------------------------------------------------------------
    # Message log
    # -------------------------------------------------------------------------

    print("\n" + "-" * 60)
    print("  GRAPH EXECUTION LOG")
    print("-" * 60)

    for msg in result["messages"]:
        print(f"  {msg}")


    return result



def wants_to_exit(value: str) -> bool:

    return value.strip().lower() in (
        "quit",
        "exit",
        "q"
    )



if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  AI DESTINATION PLANNER")
    print("=" * 60)

    print(
        "\n  Tell me about your trip and I'll create "
        "a personalized itinerary."
    )

    print(
        "  Type 'quit', 'exit', or 'q' at any time to exit.\n"
    )


    while True:

        # ---------------------------------------------------------------------
        # Destination
        # ---------------------------------------------------------------------

        destination = input(
            "  Where would you like to travel? > "
        ).strip()

        if wants_to_exit(destination):

            print(
                "\n  No problem. Happy travels!\n"
            )

            break

        if not destination:

            print(
                "\n  Please enter a destination.\n"
            )

            continue


        # ---------------------------------------------------------------------
        # Duration
        # ---------------------------------------------------------------------

        duration_input = input(
            "  How many days is your trip? > "
        ).strip()

        if wants_to_exit(duration_input):

            print(
                "\n  No problem. Happy travels!\n"
            )

            break

        try:

            duration_days = int(duration_input)

            if duration_days <= 0:
                print(
                    "\n  Please enter a number greater than 0.\n"
                )
                continue

        except ValueError:

            print(
                "\n  Please enter the duration as a number.\n"
            )

            continue


        # ---------------------------------------------------------------------
        # Preferences
        # ---------------------------------------------------------------------

        preferences = input(
            "  What are you interested in? "
            "(food, history, nature, nightlife, etc.) > "
        ).strip()

        if wants_to_exit(preferences):

            print(
                "\n  No problem. Happy travels!\n"
            )

            break

        if not preferences:

            print(
                "\n  Please enter at least one travel preference.\n"
            )

            continue


        # ---------------------------------------------------------------------
        # Budget
        # ---------------------------------------------------------------------

        budget = input(
            "  What is your approximate budget? > "
        ).strip()

        if wants_to_exit(budget):

            print(
                "\n  No problem. Happy travels!\n"
            )

            break

        if not budget:

            print(
                "\n  Please enter an approximate budget.\n"
            )

            continue


        # ---------------------------------------------------------------------
        # Run graph
        # ---------------------------------------------------------------------

        run_trip_planner(
            destination=destination,
            duration_days=duration_days,
            preferences=preferences,
            budget=budget
        )


        # ---------------------------------------------------------------------
        # Ask whether user wants another itinerary
        # ---------------------------------------------------------------------

        print("\n" + "=" * 60)

        again = input(
            "\n  Would you like to plan another trip? (yes/no) > "
        ).strip().lower()

        if again not in (
            "yes",
            "y"
        ):

            print(
                "\n  Happy travels! Goodbye!\n"
            )

            break

        print("\n")