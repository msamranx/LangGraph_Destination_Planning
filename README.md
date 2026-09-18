# AI Destination Planner with LangGraph

A beginner-friendly **LangGraph project** that demonstrates how to build a multi-node AI workflow for personalized travel planning.

The application takes a user's **destination, trip duration, travel preferences, and budget** and uses multiple specialist nodes to generate a personalized day-by-day itinerary.

The project demonstrates several important LangGraph concepts, including **state management, parallel execution, fan-in, decision nodes, and conditional routing**.

---

## What You'll Learn

This project demonstrates how to:

* Define shared state using **Pydantic**
* Create specialized LangGraph nodes
* Run multiple AI tasks in **parallel**
* Combine parallel outputs using **fan-in**
* Use an LLM as a **decision node**
* Implement **conditional routing**
* Generate different outputs depending on graph state
* Maintain a simple graph execution log
* Gracefully exit a command-line application

---

## How It Works

The user provides four pieces of information:

* **Destination**
* **Trip duration**
* **Travel preferences**
* **Approximate budget**

Three specialist nodes then work in parallel.

### 1. `plan_attractions`

Suggests sightseeing locations and experiences based on the user's destination, interests, trip duration, and budget.

### 2. `plan_food_and_culture`

Suggests local food, markets, neighborhoods, cultural activities, and destination-specific experiences.

### 3. `plan_trip_logistics`

Considers:

* Geographic grouping
* Transportation strategy
* Daily pacing
* Travel between areas
* Budget considerations
* Avoiding unnecessary backtracking

Once all three specialists finish, their recommendations are passed to the decision node.

### 4. `select_itinerary_style`

Analyzes the user profile and specialist recommendations and selects one of two itinerary styles:

* **Relaxed** — fewer activities, slower pacing, and more free time
* **Packed** — fuller days with more sightseeing and experiences

LangGraph then uses a **conditional edge** to route execution to the appropriate final node.

### 5. Final Itinerary

Depending on the decision, one of the following runs:

* `relaxed_itinerary`
* `activity_packed_itinerary`

The selected node combines all specialist recommendations into the final day-by-day itinerary.

---

## Graph Structure

```text
                         START
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
 plan_attractions   plan_food_and_culture  plan_trip_logistics
          |                |                |
          +----------------+----------------+
                           |
                           v
                select_itinerary_style
                           |
                    (conditional)
                     /           \
                    /             \
               relaxed           packed
                  |                |
                  v                v
        relaxed_itinerary   activity_packed_itinerary
                  |                |
                  v                v
                 END              END
```

The three specialist nodes branch from `START`, allowing LangGraph to execute them in parallel.

Their results then **fan in** to `select_itinerary_style`.

The decision stored in `itinerary_style` determines which final node runs.

---

## Project Structure

```text
destination-planner/
│
├── destination_planning_graph.py
├── .env
├── requirements.txt
└── README.md
```

---

## Requirements

* Python 3.10+
* OpenAI API key
* LangGraph
* LangChain OpenAI
* Pydantic
* python-dotenv

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd destination-planner
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langgraph langchain-openai python-dotenv pydantic
```

Alternatively, create a `requirements.txt` containing:

```text
langgraph
langchain-openai
python-dotenv
pydantic
```

Then run:

```bash
pip install -r requirements.txt
```

---

## Configure the OpenAI API Key

Create a `.env` file in the project directory:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

The application loads the environment variables using:

```python
from dotenv import load_dotenv

load_dotenv()
```

> **Important:** Never commit your `.env` file or API key to GitHub.

Add `.env` to your `.gitignore`:

```text
.env
.venv/
__pycache__/
```

---

## Running the Application

Run:

```bash
python destination_planning_graph.py
```

The application will ask for your trip details:

```text
============================================================
  AI DESTINATION PLANNER
============================================================

  Tell me about your trip and I'll create a personalized itinerary.
  Type 'quit', 'exit', or 'q' at any time to exit.

  Where would you like to travel? > Vietnam
  How many days is your trip? > 10
  What are you interested in? (food, history, nature, nightlife, etc.) > food, culture, nature
  What is your approximate budget? > ₹150000
```

The graph will then run the specialist planners, determine an itinerary style, and generate the final itinerary.

---



This learning project can be extended with features such as:

* Live weather information
* Maps and geographic APIs
* Real transportation data
* Hotel recommendations
* Restaurant recommendations
* Attraction opening hours
* Current pricing
* Web search
* Structured LLM outputs
* User preference memory
* Human approval before generating the final itinerary
* Itinerary export to PDF
* Interactive web interface using Streamlit or Gradio
* LangGraph checkpointing and persistence

---

## License

This project is intended for educational and learning purposes. Add your preferred open-source license before distributing or publishing the project.
