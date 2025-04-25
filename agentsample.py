from openai import AzureOpenAI
import os
from web_agent import search_bing  # Assuming you have a proper search function
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Tuple, Union
from typing_extensions import TypedDict
import streamlit as st
import operator
plan_step, execute_step, replan_step = "", "", ""
from dotenv import load_dotenv
load_dotenv()

st.title("Agentic Research Assistant")
# Azure OpenAI client setup
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-01-preview",
)

session_state = st.session_state

if "plan" not in session_state:
    session_state["plan"] = []

# with st.sidebar:
#     st.header("📋 Current Research Plan")
#     if "plan" in session_state and session_state["plan"]:
#         for idx, step in enumerate(session_state["plan"], 1):
#             st.markdown(f"**Step {idx}:** {step}")
#     else:
#         st.write("Plan will appear here after planning.")

# Azure Chat wrapper
class AzureChatWrapper:
    def __init__(self, client, deployment_name):
        self.client = client
        self.deployment = deployment_name

    def invoke(self, inputs):
        messages = inputs["messages"]
        formatted = [{"role": role, "content": content} for role, content in messages]
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=formatted
        )
        return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

llm = AzureChatWrapper(client, deployment_name="gpt-4.1")

# Types
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: str

class Plan(BaseModel):
    steps: List[str]

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan]


# Prompts
planner_prompt = """You are a finance research agent working in Oct 2024. For the given objective, come up with a simple step-by-step plan. 
This plan should involve individual tasks that, if executed correctly, will yield the correct answer. 
Do not add any superfluous steps. Make sure that each step has all the information needed."""

# Planning step
def plan_step(state: PlanExecute):
    query = state["input"]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
            {"role": "user", "content": f"{planner_prompt}\n\nObjective: {query}"}
        ]
    )
    plan_text = response.choices[0].message.content
    steps = [step.strip() for step in plan_text.split("\n") if step.strip()]
    return {"plan": steps}

def execute_step(state: PlanExecute):
    plan = state["plan"]
    task = plan[0]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task_prompt = f"For the following plan:\n{plan_str}\n\nYou are tasked with executing step 1: {task}."
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
            {"role": "user", "content": task_prompt}
        ]
    )
    result = response.choices[0].message.content
    return {"past_steps": [(task, result)]}


# Replan step
def replan_step(state: PlanExecute):
    remaining_steps = state["plan"][1:]  # remove the executed one
    if not remaining_steps:
        # If no more steps, summarize and return
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
                {"role": "user", "content": f"Given the steps already taken:\n{state['past_steps']}\nWhat is the final recommendation?"}
            ]
        )
        return {"response": response.choices[0].message.content}
    else:
        return {"plan": remaining_steps}

# Control flow
def should_end(state: PlanExecute):
    return END if "response" in state and state["response"] else "agent"
# Workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])
app = workflow.compile()

# Inputs
inputs = {
    "input": "Should we invest in Tesla given the current situation of EV?",
    "plan": [],
    "past_steps": [],
    "response": ""
}

def run_sync_app(Topic: str):
    
    state = {
        "input": Topic,
        "plan": [],
        "past_steps": [],
        "response": ""
    }
    planner_prompt = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Describe the qualities and domain expertise of a research assistant best suited for the following research topic:\n\n{Topic}\n\nRespond only with a short paragraph."}
            ]
        ).choices[0].message.content

    current_node = START
    result = {}

    while current_node != END:
        if current_node == START:
            current_node = "planner"
            result = plan_step(state)
            session_state["plan"] = result.get("plan", session_state.get("plan", []))

            # st.write(result)

        elif current_node == "planner":
            state.update(result)
            result = execute_step(state)
            session_state["plan"] = result.get("plan", session_state.get("plan", []))

            current_node = "agent"
            # st.write(result)


        elif current_node == "agent":
            # Merge past steps
            past = state.get("past_steps", [])
            new_step = result.get("past_steps", [])
            state["past_steps"] = past + new_step
            state["plan"] = state["plan"][1:]  # remove completed step
            result = replan_step(state)
            session_state["plan"] = result.get("plan", state["plan"])
            
            # 🔽 ADD DISPLAY BLOCK HERE
            # with st.sidebar:
            #     st.subheader("✅ Executed Steps")
            #     for i, (task, result_text) in enumerate(state["past_steps"], 1):
            #         st.markdown(f"**Step {i}:** {task}\n\n_Result:_ {result_text}")
            
            current_node = "replan"


        elif current_node == "replan":
            state.update(result)
            current_node = should_end(state)

    st.write("\n✅ Final Output:\n")
    st.write(state["response"])
    st.markdown(session_state["plan"])
    # with st.sidebar:
    #     st.subheader("✅ Executed Steps")
    #     for i, (task, result_text) in enumerate(state["past_steps"], 1):
    #         st.markdown(f"**Step {i}:** {task}\n\n_Result:_ {result_text}")

if __name__ == "__main__":
    Topic = st.text_area("Research Topic")
    if st.button("Run Research"):
        if Topic.strip():  # Only run if there's valid input
            run_sync_app(Topic)
            
        else:
            st.warning("Please enter a research topic.")
