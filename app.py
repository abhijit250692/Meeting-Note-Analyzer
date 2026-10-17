import os
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError


load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# -------------------
# State
# -------------------
class GraphState(TypedDict):
    input: str
    topics: str
    summary: str
    actionItems: str
    priority: str
    minutes: str


# -------------------
# Input Agent
# -------------------
def input_agent(state):
    return


# -------------------
# Key Topics Extractor Agent
# -------------------
def key_topics_extractor_agent(state):
    
    prompt = f"""
    Extract the key topics discussed in the meeting {state['input']}. 
    List the main topics in bullet points.
    """
    topics = llm.invoke(prompt).content.strip()
    return {"topics": topics}


# -------------------
# Summary Extractor Agent
# -------------------
def summary_extractor_agent(state):
    
    prompt = f"""
    Generate a short summary of the meeting using meeting context {state['input']} explaining what the meeting was about. 
    The summary should be concise and limited to 3-5 sentences max. Do not hallucinate.
    """
    summary = llm.invoke(prompt).content.strip()
    return {"summary": summary}


# -------------------
# Action Items Identifier Agent
# -------------------
def action_items_identifier_agent(state):
    
    prompt = f"""
    Identify the action items discussed in the meeting {state['input']} along with the responsible person for each action item.
    If the owner is not mentioned, label it as 'Owner: Not specified'.
    If no action items detected, return 'No action items identified in this meeting'.
    Else list all the action items in bullet points.
    """
    action_items = llm.invoke(prompt).content.strip()
    return {"actionItems": action_items}


# -------------------
# Priority Detector Agent
# -------------------
def priority_detector_agent(state):
    
    prompt = f"""
    Identify the priority level including the action items discussed in the meeting {state['input']}.
    
    Possible outputs include:
    - High
    - Medium
    - Low

    Output should be a single word from possible output options.
    Detect urgency from phrases like 'urgent', 'ASAP', 'by tomorrow', 'this week' etc.
    """
    priority = llm.invoke(prompt).content.strip()
    return {"priority": priority}


# -------------------
# Minutes Generator Agent
# -------------------
def minutes_generator_agent(state):
    prompt = f"""
    Generate meeting minutes based on the details available in the state {state}. 
    The minutes should be in a structured format which include only below fields.

    1. Meeting Summary
    2. Key Topics
    3. Action Items & Task Owners
    4. Priority Level

    """
    minutes = llm.invoke(prompt).content
    return {"minutes": minutes}


# -------------------
# Router used to determine next node in workflow
# -------------------
def priority_Check_router(state):
    if state['actionItems'] == 'No action items identified in this meeting':
        return "Priority Check Not Required"
    else:
        return "Priority Check Required"



def build_meeting_workflow():

    workflow = StateGraph(GraphState)
    workflow.add_node("input", input_agent)
    workflow.add_node("key_topics_extractor", key_topics_extractor_agent)
    workflow.add_node("summary_extractor", summary_extractor_agent)
    workflow.add_node("action_items_identifier", action_items_identifier_agent)
    workflow.add_node("priority_detector", priority_detector_agent)
    workflow.add_node("minutes_generator", minutes_generator_agent)

    workflow.add_edge(START, "input")
    workflow.add_edge("input", "key_topics_extractor")
    workflow.add_edge("input", "summary_extractor")
    workflow.add_edge("key_topics_extractor", "action_items_identifier")
    workflow.add_edge("summary_extractor", "action_items_identifier")

    workflow.add_conditional_edges(
        "action_items_identifier",
        priority_Check_router,
        {
            "Priority Check Required": "priority_detector",
            "Priority Check Not Required": "minutes_generator"
        }
    )

    workflow.add_edge("priority_detector", "minutes_generator")
    workflow.add_edge("minutes_generator", END)

    return workflow.compile()
    

st.set_page_config(page_title="Meeting Note Analyzer", page_icon="📝", layout="wide")
st.title("Meeting Note Analyzer")
st.caption(
    "Turn a meeting transcript into structured minutes."
)

with st.form("meeting_transcript_form"):
    transcript = st.text_area(
        "Meeting transcript",
        height=260,
        placeholder=(
            "Paste a meeting transcript"
        )
    )
    submitted = st.form_submit_button("Analyze meeting", type="primary")

if submitted:
    st.session_state.pop("meeting_result", None)
    if not transcript.strip():
        st.error("Enter a meeting transcript before starting the analysis.")
    else:
        try:
            with st.spinner("Analyzing the meeting transcript..."):
                st.session_state["meeting_result"] = build_meeting_workflow().invoke(
                    {"input": transcript.strip()}
                )
        except (OpenAIError, ValueError) as error:
            st.error(f"Meeting analysis failed: {error}")

result = st.session_state.get("meeting_result")
if result:
    st.markdown(result["minutes"])