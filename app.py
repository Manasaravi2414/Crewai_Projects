"""
CrewAI Projects API — all 10 assignment projects.

Run:
    pip install -r requirements.txt
    # Create .env with GEMINI_API_KEY=your_key
    uvicorn app:app --reload

Each endpoint accepts:
    {"input": "your task or source text"}

Projects 1–4 use CrewAI Crews. Projects 5–10 use CrewAI Flows.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start


# ---------------------------------------------------------------------------
# 1. Application setup and shared request/response models
# ---------------------------------------------------------------------------

load_dotenv()

app = FastAPI(
    title="CrewAI Projects API",
    description="Ten CrewAI projects exposed through /agent1 to /agent10.",
    version="1.0.0",
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AgentRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Prompt or source content")


class AgentResponse(BaseModel):
    output: str


class WorkflowState(BaseModel):
    """Shared state object used by the six Flow projects."""
    user_input: str = ""
    category: str = ""
    analysis: str = ""
    output: str = ""
    approval: str = ""
    revision_count: int = 0
    linkedin_output: str = ""
    twitter_output: str = ""
    newsletter_output: str = ""


def get_llm() -> LLM:
    """Create the Gemini LLM; fail with a helpful message if the key is absent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing. Add it to your .env file.",
        )

    # Override GEMINI_MODEL in .env if your account uses another supported model.
    model_name = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")
    return LLM(model=model_name, api_key=api_key, temperature=0.2)


def make_agent(
    llm: LLM,
    role: str,
    goal: str,
    backstory: str,
    tools: list | None = None,
) -> Agent:
    """Keep agent definitions consistent across projects."""
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        verbose=False,
        allow_delegation=False,
    )


def run_crew(
    agents: list[Agent],
    tasks: list[Task],
    process=Process.sequential,
    inputs: dict[str, Any] | None = None,
    manager_agent: Agent | None = None,
) -> str:
    """Run one Crew and convert its result to a plain string."""
    crew_options: dict[str, Any] = {
        "agents": agents,
        "tasks": tasks,
        "process": process,
        "verbose": False,
    }
    if manager_agent is not None:
        crew_options["manager_agent"] = manager_agent
    crew = Crew(**crew_options)
    result = crew.kickoff(inputs=inputs or {})
    return str(result)


def save_markdown(filename: str, heading: str, content: str) -> str:
    """Save a readable project artifact and return its text."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    path = OUTPUT_DIR / safe_name
    path.write_text(f"# {heading}\n\n{content}\n", encoding="utf-8")
    return f"{content}\n\nSaved artifact: {path.as_posix()}"


def run_flow(flow_class: type[Flow], user_input: str) -> str:
    """Instantiate a Flow, seed its state, execute it, and return the result."""
    flow = flow_class()
    flow.state.user_input = user_input
    result = flow.kickoff()
    # Most methods return the final text. State output is a safe fallback.
    if result is None or not str(result).strip():
        result = flow.state.output
    return str(result)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "CrewAI Projects API is running.",
        "docs": "/docs",
        "health": "/health",
        "projects": "Use POST /agent1 through POST /agent10.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# PROJECT 1 — Blog Post Crew (sequential process)
# ---------------------------------------------------------------------------

@app.post("/agent1", response_model=AgentResponse)
def blog_post_crew(request: AgentRequest) -> AgentResponse:
    """Research a topic and turn the findings into a blog post."""
    try:
        llm = get_llm()
        researcher = make_agent(
            llm, "Blog Researcher",
            "Gather accurate, useful points about the requested topic.",
            "You are careful about separating established facts from assumptions.",
        )
        writer = make_agent(
            llm, "Blog Writer",
            "Write a clear, engaging, well-structured blog post.",
            "You write for curious readers and explain unfamiliar terms simply.",
        )
        research_task = Task(
            description=(
                "Research the topic below using your knowledge. Provide a concise "
                "outline, key facts, examples, caveats, and useful subheadings.\n"
                "Topic: {topic}"
            ),
            expected_output="A factual research brief with an outline and key points.",
            agent=researcher,
        )
        writing_task = Task(
            description=(
                "Use the research brief to write a polished 700–900 word blog post. "
                "Include a title, introduction, descriptive headings, examples, and "
                "a short conclusion. Do not invent citations or claim web research "
                "was performed if it was not.\nTopic: {topic}"
            ),
            expected_output="A complete blog post in Markdown.",
            agent=writer,
            context=[research_task],
        )
        output = run_crew(
            [researcher, writer],
            [research_task, writing_task],
            inputs={"topic": request.input},
        )
        return AgentResponse(output=save_markdown("blog_post.md", "Blog Post", output))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent1] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Blog Post Crew failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 2 — Competitor Analysis Crew (hierarchical process)
# ---------------------------------------------------------------------------

@app.post("/agent2", response_model=AgentResponse)
def competitor_analysis_crew(request: AgentRequest) -> AgentResponse:
    """
    The input should identify a business/industry and ideally competitor URLs.
    This implementation clearly labels knowledge-based analysis; it does not
    silently pretend that live websites were fetched.
    """
    try:
        llm = get_llm()
        analyst = make_agent(
            llm, "Competitor Research Analyst",
            "Identify competitor positioning, likely strengths, weaknesses, and gaps.",
            "You produce structured competitive research and state uncertainty clearly.",
        )
        pricing = make_agent(
            llm, "Pricing Analyst",
            "Compare pricing models and explain their strategic implications.",
            "You never invent exact current prices; mark unknowns as needing verification.",
        )
        strategist = make_agent(
            llm, "Competitive Strategy Consultant",
            "Synthesize research into practical, prioritized recommendations.",
            "You turn market observations into actionable business decisions.",
        )
        manager = make_agent(
            llm, "Research Project Manager",
            "Coordinate the specialist analyses into one consistent report.",
            "You resolve overlap, ask for evidence, and produce a coherent final report.",
        )

        research_task = Task(
            description=(
                "Analyze the following business, industry, and competitor details. "
                "Identify the likely direct and indirect competitors, target users, "
                "positioning, differentiators, and evidence gaps. Distinguish facts "
                "in the input from reasonable hypotheses.\nInput: {details}"
            ),
            expected_output="Competitor landscape and positioning analysis.",
            agent=analyst,
        )
        pricing_task = Task(
            description=(
                "Create a pricing comparison framework for the competitors named in "
                "the input. Compare free tiers, subscriptions, usage-based pricing, "
                "enterprise plans, and value metrics where relevant. If actual prices "
                "are not supplied, write 'verify current pricing' instead of guessing.\n"
                "Input: {details}"
            ),
            expected_output="Pricing comparison with unknowns explicitly flagged.",
            agent=pricing,
        )
        strategy_task = Task(
            description=(
                "Using the competitor and pricing analyses, create a practical report "
                "with: executive summary, competitor table, SWOT-style observations, "
                "market gaps, 3–5 recommendations, risks, and next steps.\nInput: {details}"
            ),
            expected_output="A concise, decision-ready competitor analysis report.",
            agent=strategist,
            context=[research_task, pricing_task],
        )
        output = run_crew(
            [analyst, pricing, strategist],  # Do not include manager here
            [research_task, pricing_task, strategy_task],
            process=Process.hierarchical,
            inputs={"details": request.input},
            manager_agent=manager,
        )
        return AgentResponse(
            output=save_markdown("competitor_analysis.md", "Competitor Analysis", output)
        )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent2] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Competitor Analysis Crew failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 3 — Meeting Notes Crew (parallel analysis tasks + formatter)
# ---------------------------------------------------------------------------

@app.post("/agent3", response_model=AgentResponse)
def meeting_notes_crew(request: AgentRequest) -> AgentResponse:
    """Convert pasted meeting transcript/rough notes into organized meeting notes."""
    try:
        llm = get_llm()
        summarizer = make_agent(
            llm, "Meeting Summarizer",
            "Summarize the meeting accurately and concisely.",
            "You preserve decisions and important context without adding facts.",
        )
        action_finder = make_agent(
            llm, "Action Item Extractor",
            "Extract actions, owners, and deadlines from meeting notes.",
            "You mark missing owners or deadlines as 'Not specified'.",
        )
        decision_finder = make_agent(
            llm, "Decision Logger",
            "Identify explicit decisions, unresolved questions, and risks.",
            "You distinguish a final decision from a suggestion or discussion.",
        )
        formatter = make_agent(
            llm, "Meeting Notes Formatter",
            "Combine analyses into clean meeting minutes.",
            "You produce readable Markdown with consistent headings and tables.",
        )

        summary_task = Task(
            description="Summarize this meeting transcript in 5–10 bullets:\n{transcript}",
            expected_output="Meeting summary bullets.",
            agent=summarizer,
            async_execution=True,
        )
        action_task = Task(
            description=(
                "Extract action items from this transcript. Return a table with "
                "Action, Owner, Deadline, and Status. Use 'Not specified' when absent.\n"
                "{transcript}"
            ),
            expected_output="An action-item table.",
            agent=action_finder,
            async_execution=True,
        )
        decision_task = Task(
            description=(
                "List confirmed decisions, open questions, blockers, and risks from "
                "this transcript. Do not infer a decision that was not made.\n{transcript}"
            ),
            expected_output="Decisions and open questions.",
            agent=decision_finder,
            async_execution=True,
        )
        format_task = Task(
            description=(
                "Combine the preceding analysis into meeting minutes with sections "
                "Summary, Decisions, Action Items, Open Questions, and Next Meeting. "
                "Use 'Not specified' if a detail is missing.\nTranscript:\n{transcript}"
            ),
            expected_output="Complete meeting minutes in Markdown.",
            agent=formatter,
            context=[summary_task, action_task, decision_task],
        )
        output = run_crew(
            [summarizer, action_finder, decision_finder, formatter],
            [summary_task, action_task, decision_task, format_task],
            inputs={"transcript": request.input},
        )
        return AgentResponse(output=save_markdown("meeting_notes.md", "Meeting Notes", output))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent3] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Meeting Notes Crew failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 4 — Data Report Crew (pandas summary + insights and review)
# ---------------------------------------------------------------------------

def summarize_tabular_input(raw_text: str) -> str:
    """
    If input contains CSV, calculate basic pandas statistics. For non-CSV input,
    return a clear note so the LLM can still analyze the supplied description.
    """
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if len(lines) < 2 or not any("," in line for line in lines[:3]):
        return (
            "No clearly parseable CSV table was detected. Analyze the supplied "
            "description qualitatively and state that numerical statistics were unavailable."
        )
    try:
        from io import StringIO
        frame = pd.read_csv(StringIO(raw_text))
        if frame.empty:
            return "The CSV was parsed but contains no data rows."
        numeric = frame.select_dtypes(include="number")
        details = [
            f"Rows: {len(frame)}",
            f"Columns: {len(frame.columns)}",
            f"Column names: {', '.join(map(str, frame.columns))}",
            f"Missing values by column: {frame.isna().sum().to_dict()}",
        ]
        if not numeric.empty:
            details.append("Numeric summary:\n" + numeric.describe().round(3).to_string())
        else:
            details.append("No numeric columns were detected.")
        for column in frame.select_dtypes(exclude="number").columns[:5]:
            details.append(
                f"Top values in {column}: {frame[column].value_counts(dropna=False).head(5).to_dict()}"
            )
        return "\n\n".join(details)
    except Exception as exc:
        return f"CSV parsing was unsuccessful ({type(exc).__name__}). Analyze the input cautiously."


@app.post("/agent4", response_model=AgentResponse)
def data_report_crew(request: AgentRequest) -> AgentResponse:
    """Accept CSV text or a description of a dataset."""
    try:
        llm = get_llm()
        stats = summarize_tabular_input(request.input)
        analyst = make_agent(
            llm, "Data Analyst",
            "Explain dataset patterns and derive evidence-based insights.",
            "You use the provided statistics and never fabricate numeric results.",
        )
        reviewer = make_agent(
            llm, "Data Report Reviewer",
            "Check the report for clarity, unsupported claims, and useful caveats.",
            "You are rigorous about data limitations and distinguish correlation from causation.",
        )
        analysis_task = Task(
            description=(
                "Analyze the dataset input and the pandas summary below. Provide data "
                "quality observations, trends, notable distributions, possible business "
                "implications, and limitations. Do not invent statistics.\n"
                "Original input:\n{dataset}\n\nPandas summary:\n{stats}"
            ),
            expected_output="Evidence-based dataset analysis draft.",
            agent=analyst,
        )
        review_task = Task(
            description=(
                "Review and improve the analysis. Include Executive Summary, Dataset "
                "Overview, Key Findings, Data Quality, Recommendations, and Limitations. "
                "Flag unsupported conclusions.\nDataset:\n{dataset}\n\nStatistics:\n{stats}"
            ),
            expected_output="Reviewed data report in Markdown.",
            agent=reviewer,
            context=[analysis_task],
        )
        # Crew task interpolation uses the kickoff inputs below.
        crew = Crew(
            agents=[analyst, reviewer],
            tasks=[analysis_task, review_task],
            process=Process.sequential,
            verbose=False,
        )
        output = str(crew.kickoff(inputs={"dataset": request.input, "stats": stats}))
        return AgentResponse(output=save_markdown("data_report.md", "Data Report", output))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent4] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Data Report Crew failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# Shared Flow helper
# ---------------------------------------------------------------------------

def flow_agent(role: str, goal: str, backstory: str) -> Agent:
    return make_agent(get_llm(), role, goal, backstory)


def ask_agent(agent: Agent, description: str, expected_output: str, inputs: dict[str, Any]) -> str:
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff(inputs=inputs))


# ---------------------------------------------------------------------------
# PROJECT 5 — Customer Support Flow (router categories)
# ---------------------------------------------------------------------------

class CustomerSupportFlow(Flow[WorkflowState]):
    @start()
    def classify_request(self) -> str:
        text = self.state.user_input.lower()
        if any(word in text for word in ("refund", "payment", "charged", "invoice", "billing", "subscription")):
            self.state.category = "billing"
        elif any(word in text for word in ("bug", "error", "crash", "not working", "broken", "technical")):
            self.state.category = "technical"
        elif any(word in text for word in ("cancel", "return", "delivery", "shipping", "order", "account")):
            self.state.category = "account_or_order"
        else:
            self.state.category = "general"
        return self.state.category

    @router(classify_request)
    def route_request(self):
        return self.state.category

    @listen("billing")
    def handle_billing(self) -> str:
        agent = flow_agent("Billing Support Specialist", "Resolve billing questions helpfully.",
                           "You explain billing steps and never request passwords or full card details.")
        return ask_agent(agent, "Draft a helpful billing support reply for this request:\n{message}",
                         "A clear customer-facing billing reply.", {"message": self.state.user_input})

    @listen("technical")
    def handle_technical(self) -> str:
        agent = flow_agent("Technical Support Specialist", "Troubleshoot technical problems safely.",
                           "You ask for relevant non-sensitive details and offer ordered troubleshooting steps.")
        return ask_agent(agent, "Draft a technical support reply with prioritized troubleshooting steps:\n{message}",
                         "A safe, step-by-step troubleshooting response.", {"message": self.state.user_input})

    @listen("account_or_order")
    def handle_account_order(self) -> str:
        agent = flow_agent("Account and Order Support Specialist", "Help with account and order issues.",
                           "You give clear next steps and do not claim to access private account systems.")
        return ask_agent(agent, "Draft a support reply about this account/order issue:\n{message}",
                         "A helpful account or order support response.", {"message": self.state.user_input})

    @listen("general")
    def handle_general(self) -> str:
        agent = flow_agent("General Customer Support Specialist", "Answer general customer questions.",
                           "You are polite, concise, and transparent about uncertainty.")
        return ask_agent(agent, "Draft a helpful general customer support response:\n{message}",
                         "A concise customer-facing support response.", {"message": self.state.user_input})

    @listen(or_(handle_billing, handle_technical, handle_account_order, handle_general))
    def finalize(self, response: str) -> str:
        self.state.output = f"Category: {self.state.category}\n\n{response}"
        return self.state.output


@app.post("/agent5", response_model=AgentResponse)
def customer_support_flow(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(CustomerSupportFlow, request.input))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent5] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Customer Support Flow failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 6 — Resume Screening Flow (score and shortlist/reject)
# ---------------------------------------------------------------------------

class ResumeScreeningFlow(Flow[WorkflowState]):
    @start()
    def score_resume(self) -> str:
        agent = flow_agent(
            "Resume Screening Analyst",
            "Compare resume evidence against the job criteria and produce a transparent score.",
            "You evaluate job-related qualifications only, avoid protected characteristics, "
            "and identify evidence and missing information.",
        )
        self.state.analysis = ask_agent(
            agent,
            "Evaluate this resume against the role description. Score 0–100 using job-related "
            "skills, relevant experience, and required qualifications. Explain evidence and "
            "gaps. Do not use age, gender, ethnicity, disability, or other protected traits. "
            "If role criteria are absent, state assumptions.\nResume and role:\n{material}",
            "Score, evidence, gaps, and caveats.",
            {"material": self.state.user_input},
        )
        match = re.search(r"(?<!\d)(\d{1,3})\s*/\s*100", self.state.analysis)
        if not match:
            match = re.search(r"(?i)score[^0-9]{0,20}(\d{1,3})", self.state.analysis)
        score = int(match.group(1)) if match else 0
        self.state.category = "shortlist" if score >= 70 else "review"
        return self.state.category

    @router(score_resume)
    def route_resume(self):
        return self.state.category

    @listen("shortlist")
    def create_shortlist_recommendation(self) -> str:
        self.state.output = (
            "RECOMMENDATION: SHORTLIST (score threshold: 70/100)\n\n"
            + self.state.analysis
        )
        return self.state.output

    @listen("review")
    def review_manually(self) -> str:
        self.state.output = (
            "RECOMMENDATION: HUMAN REVIEW REQUIRED (score not confidently above 70/100)\n\n"
            + self.state.analysis
        )
        return self.state.output


@app.post("/agent6", response_model=AgentResponse)
def resume_screening_flow(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(ResumeScreeningFlow, request.input))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent6] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Resume Screening Flow failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 7 — Bug Triage Flow (bug / feature / question router)
# ---------------------------------------------------------------------------

class BugTriageFlow(Flow[WorkflowState]):
    @start()
    def classify_ticket(self) -> str:
        text = self.state.user_input.lower()
        if any(word in text for word in ("bug", "crash", "exception", "error", "fails", "broken", "defect")):
            self.state.category = "bug"
        elif any(word in text for word in ("feature", "add support", "would like", "enhancement", "request")):
            self.state.category = "feature"
        else:
            self.state.category = "question"
        return self.state.category

    @router(classify_ticket)
    def route_ticket(self):
        return self.state.category

    @listen("bug")
    def triage_bug(self) -> str:
        agent = flow_agent("Bug Triage Engineer", "Prioritize bugs and identify diagnostic information.",
                           "You avoid assuming root cause without logs or reproduction steps.")
        self.state.output = ask_agent(
            agent, "Triage this bug report. Provide severity (with rationale), impact, "
            "reproduction questions, suspected area only if justified, and next steps:\n{ticket}",
            "Structured bug triage.", {"ticket": self.state.user_input})
        return self.state.output

    @listen("feature")
    def triage_feature(self) -> str:
        agent = flow_agent("Feature Request Analyst", "Clarify feature value and acceptance criteria.",
                           "You focus on user outcomes, scope, dependencies, and measurable acceptance criteria.")
        self.state.output = ask_agent(
            agent, "Analyze this feature request. Provide user value, scope, acceptance criteria, "
            "dependencies, and questions for the requester:\n{ticket}",
            "Structured feature assessment.", {"ticket": self.state.user_input})
        return self.state.output

    @listen("question")
    def triage_question(self) -> str:
        agent = flow_agent("Issue Support Analyst", "Answer questions or request missing context.",
                           "You are helpful and do not label a question as a bug without evidence.")
        self.state.output = ask_agent(
            agent, "Respond to this product/technical question. Answer if possible; otherwise "
            "list the missing details needed:\n{ticket}",
            "A helpful answer or clarification request.", {"ticket": self.state.user_input})
        return self.state.output


@app.post("/agent7", response_model=AgentResponse)
def bug_triage_flow(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(BugTriageFlow, request.input))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent7] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Bug Triage Flow failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 8 — Content Repurposing Flow (parallel outputs + join)
# ---------------------------------------------------------------------------

class ContentRepurposingFlow(Flow[WorkflowState]):
    @start()
    def prepare_source(self) -> str:
        agent = flow_agent("Content Strategist", "Identify the key message and audience from source content.",
                           "You preserve the source's meaning and do not introduce unsupported claims.")
        self.state.analysis = ask_agent(
            agent, "Extract the main message, audience, 3–6 key points, and tone from this content:\n{source}",
            "A compact content brief.", {"source": self.state.user_input})
        return self.state.analysis

    @listen(prepare_source)
    def create_linkedin(self, brief: str) -> str:
        agent = flow_agent("LinkedIn Writer", "Create a professional LinkedIn post.",
                           "You write with a strong opening, useful insight, and restrained hashtags.")
        return ask_agent(agent, "Turn this brief into a LinkedIn post of 120–220 words. Keep it authentic.\n{brief}",
                         "One LinkedIn post.", {"brief": brief})

    @listen(prepare_source)
    def create_twitter(self, brief: str) -> str:
        agent = flow_agent("Short-Form Social Writer", "Create concise posts for X/Twitter.",
                           "You write concise copy and preserve the central idea.")
        return ask_agent(agent, "Create 3 alternative X/Twitter posts, each under 280 characters.\n{brief}",
                         "Three short posts.", {"brief": brief})

    @listen(prepare_source)
    def create_newsletter(self, brief: str) -> str:
        agent = flow_agent("Newsletter Writer", "Create a short newsletter version of content.",
                           "You write a useful subject line, preview, and reader-friendly body.")
        return ask_agent(agent, "Create a newsletter subject, preview text, and 150–250 word body.\n{brief}",
                         "A newsletter draft.", {"brief": brief})

    @listen(or_(create_linkedin, create_twitter, create_newsletter))
    def combine_versions(self, result: str) -> str:
        # Each branch is a separate Flow event. Return each finished version;
        # the Flow runtime may invoke this listener for each branch result.
        current = str(result)
        if current not in self.state.output:
            self.state.output += ("\n\n---\n\n" if self.state.output else "") + current
        return self.state.output


@app.post("/agent8", response_model=AgentResponse)
def content_repurposing_flow(request: AgentRequest) -> AgentResponse:
    try:
        result = run_flow(ContentRepurposingFlow, request.input)
        return AgentResponse(output=result)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent8] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Content Repurposing Flow failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 9 — Lead Qualification Flow (enrich and hot/warm/cold route)
# ---------------------------------------------------------------------------

class LeadQualificationFlow(Flow[WorkflowState]):
    @start()
    def enrich_lead(self) -> str:
        agent = flow_agent("Lead Research Analyst", "Extract lead needs and qualification signals.",
                           "You work only from provided details, label assumptions, and do not invent company facts.")
        self.state.analysis = ask_agent(
            agent, "Analyze this lead. Extract company, role, need, budget, authority, urgency, "
            "fit, missing data, and recommended follow-up questions. Then suggest a 0–100 "
            "qualification score with rationale.\nLead:\n{lead}",
            "Lead enrichment with an explicit numeric score.", {"lead": self.state.user_input})
        match = re.search(r"(?i)(?:score|qualification)[^0-9]{0,25}(\d{1,3})", self.state.analysis)
        score = int(match.group(1)) if match else 50
        if score >= 75:
            self.state.category = "hot"
        elif score >= 45:
            self.state.category = "warm"
        else:
            self.state.category = "cold"
        return self.state.category

    @router(enrich_lead)
    def route_lead(self):
        return self.state.category

    @listen("hot")
    def handle_hot(self) -> str:
        self.state.output = (
            "LEAD STATUS: HOT — prioritize prompt sales follow-up.\n\n" + self.state.analysis
        )
        return self.state.output

    @listen("warm")
    def handle_warm(self) -> str:
        self.state.output = (
            "LEAD STATUS: WARM — nurture and clarify qualification gaps.\n\n" + self.state.analysis
        )
        return self.state.output

    @listen("cold")
    def handle_cold(self) -> str:
        self.state.output = (
            "LEAD STATUS: COLD — use low-touch nurturing or revisit when signals change.\n\n"
            + self.state.analysis
        )
        return self.state.output


@app.post("/agent9", response_model=AgentResponse)
def lead_qualification_flow(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(LeadQualificationFlow, request.input))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent9] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Lead Qualification Flow failed. Check the server terminal.")


# ---------------------------------------------------------------------------
# PROJECT 10 — Approval Report Flow (draft -> approval/revision)
# ---------------------------------------------------------------------------

class ApprovalReportFlow(Flow[WorkflowState]):
    @start()
    def create_report(self) -> str:
        agent = flow_agent("Report Analyst", "Prepare a concise report from supplied information.",
                           "You use the supplied facts, identify assumptions, and state limitations.")
        self.state.analysis = ask_agent(
            agent, "Create a report with Executive Summary, Findings, Recommendations, Risks, "
            "and Open Questions from the following material:\n{material}",
            "A draft report in Markdown.", {"material": self.state.user_input})
        return self.state.analysis

    @router(create_report)
    def check_approval_instruction(self):
        # The endpoint remains one-request/one-response. Include a line such as
        # 'Approval: approve' or 'Approval: revise' in the input to simulate feedback.
        text = self.state.user_input.lower()
        if re.search(r"approval\s*:\s*approve\b|\bapproved\b", text):
            self.state.approval = "approved"
            return "approved"
        if re.search(r"approval\s*:\s*revise\b|\brevise\b", text):
            self.state.approval = "revise"
            return "revise"
        self.state.approval = "pending"
        return "pending"

    @listen("approved")
    def approve_report(self) -> str:
        self.state.output = "STATUS: APPROVED\n\n" + self.state.analysis
        return self.state.output

    @listen("revise")
    def revise_report(self) -> str:
        agent = flow_agent("Report Editor", "Revise a report using requested feedback.",
                           "You preserve accurate facts and clearly improve the draft based on feedback.")
        revised = ask_agent(
            agent, "Revise this report. The original user input includes the material and any "
            "revision notes. Incorporate those notes, and if they are vague, improve clarity "
            "and structure without inventing facts.\nDraft:\n{draft}\n\nOriginal input:\n{material}",
            "A revised report in Markdown.",
            {"draft": self.state.analysis, "material": self.state.user_input},
        )
        self.state.output = "STATUS: REVISED — human approval is still recommended\n\n" + revised
        return self.state.output

    @listen("pending")
    def request_approval(self) -> str:
        self.state.output = (
            "STATUS: PENDING HUMAN APPROVAL\n\n"
            "Review the draft below. To simulate the approval decision in this single-request "
            "API, include 'Approval: approve' or 'Approval: revise' in the input and submit again.\n\n"
            + self.state.analysis
        )
        return self.state.output


@app.post("/agent10", response_model=AgentResponse)
def approval_report_flow(request: AgentRequest) -> AgentResponse:
    try:
        result = run_flow(ApprovalReportFlow, request.input)
        return AgentResponse(output=save_markdown("approval_report.md", "Approval Report", result))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[agent10] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Approval Report Flow failed. Check the server terminal.")
