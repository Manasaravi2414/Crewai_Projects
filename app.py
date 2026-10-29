"""
CrewAI Multi-Agent Backend Service (Projects 1 to 10)
Exposes endpoints /agent1 to /agent10 via FastAPI.

Contract:
  POST {"input": "..."} -> {"output": "..."}
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, and_, listen, or_, router, start

load_dotenv()

app = FastAPI(
    title="CrewAI Projects API",
    description="Ten CrewAI projects exposed through /agent1 to /agent10.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# 1. Shared Schemas & Helpers
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Prompt or source content")


class AgentResponse(BaseModel):
    output: str


class ScoreResult(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Numerical score out of 100")
    rationale: str = Field(..., description="Evaluation notes and justification")


class WorkflowState(BaseModel):
    user_input: str = ""
    category: str = ""
    analysis: str = ""
    output: str = ""
    approval: str = ""


def get_llm() -> LLM:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY or GOOGLE_API_KEY is missing. Add it to your .env file.",
        )
    model_name = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")
    return LLM(model=model_name, api_key=api_key, temperature=0.2)


def make_agent(
    llm: LLM,
    role: str,
    goal: str,
    backstory: str,
    tools: list | None = None,
) -> Agent:
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


def ask_agent(agent: Agent, description: str, expected_output: str, inputs: dict[str, Any]) -> str:
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff(inputs=inputs))


def run_flow(flow_class: type[Flow], user_input: str) -> str:
    flow = flow_class()
    result = flow.kickoff(inputs={"user_input": user_input})
    if result is None or not str(result).strip():
        result = getattr(flow.state, "output", "")
    return str(result)


def save_artifact(filename: str, heading: str, content: str) -> str:
    try:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
        path = OUTPUT_DIR / safe_name
        path.write_text(f"# {heading}\n\n{content}\n", encoding="utf-8")
    except Exception:
        pass
    return content


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "CrewAI Projects API is running.",
        "routes": [f"/agent{i}" for i in range(1, 11)],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Project 1: Blog Post Crew (Sequential Process)
# ---------------------------------------------------------------------------
@app.post("/agent1", response_model=AgentResponse)
def blog_post_crew(request: AgentRequest) -> AgentResponse:
    try:
        llm = get_llm()
        researcher = make_agent(
            llm, "Senior Research Specialist",
            "Gather verified, relevant facts and an outline for {topic}.",
            "You separate established facts from speculation, organizing findings clearly.",
        )
        writer = make_agent(
            llm, "Editorial Content Writer",
            "Write a structured, engaging Markdown blog post from research.",
            "You write clear prose for an industry audience, including headings and takeaways.",
        )
        t_research = Task(
            description="Research the topic: {topic}. Provide key facts, statistics, and subheadings.",
            expected_output="Detailed research brief with bulleted key points.",
            agent=researcher,
        )
        t_write = Task(
            description="Using the research, write a 600-800 word Markdown blog post on: {topic}.",
            expected_output="Complete Markdown blog post with title, headings, and conclusion.",
            agent=writer,
            context=[t_research],
        )
        output = run_crew([researcher, writer], [t_research, t_write], inputs={"topic": request.input})
        return AgentResponse(output=save_artifact("blog_post.md", "Blog Post", output))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 1 error: {exc}")


# ---------------------------------------------------------------------------
# Project 2: Competitor Analysis Crew (Hierarchical Process)
# ---------------------------------------------------------------------------
@app.post("/agent2", response_model=AgentResponse)
def competitor_analysis_crew(request: AgentRequest) -> AgentResponse:
    try:
        llm = get_llm()
        analyst = make_agent(
            llm, "Competitive Intelligence Analyst",
            "Map competitor positioning, product capabilities, and customer segments.",
            "You analyze corporate offerings and clearly flag assumptions.",
        )
        pricing = make_agent(
            llm, "SaaS Pricing Specialist",
            "Compare subscription tiers, usage models, and monetization strategies.",
            "You compare packaging models objectively, marking unconfirmed data clearly.",
        )
        strategist = make_agent(
            llm, "Strategic Advisor",
            "Synthesize data into a comparative SWOT matrix and recommendations.",
            "You turn competitor observations into strategic action plans.",
        )
        manager = make_agent(
            llm, "Research Director",
            "Delegate and organize the competitive evaluation into a cohesive report.",
            "You oversee research depth and ensure consistent output formatting.",
        )

        t_research = Task(
            description="Examine the target business/competitors: {details}. Identify competitors and offerings.",
            expected_output="Competitor profiles and market positioning notes.",
            agent=analyst,
        )
        t_pricing = Task(
            description="Compare pricing models and tiers for competitors in: {details}.",
            expected_output="Comparative pricing breakdown.",
            agent=pricing,
        )
        t_strategy = Task(
            description="Build a comparative SWOT table and strategic takeaways for: {details}.",
            expected_output="Executive report featuring a SWOT table and recommendations in Markdown.",
            agent=strategist,
            context=[t_research, t_pricing],
        )
        output = run_crew(
            [analyst, pricing, strategist],
            [t_research, t_pricing, t_strategy],
            process=Process.hierarchical,
            inputs={"details": request.input},
            manager_agent=manager,
        )
        return AgentResponse(output=save_artifact("competitor_analysis.md", "Competitor Analysis", output))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 2 error: {exc}")


# ---------------------------------------------------------------------------
# Project 3: Meeting Notes Crew (Async Parallel Tasks + Formatter)
# ---------------------------------------------------------------------------
@app.post("/agent3", response_model=AgentResponse)
def meeting_notes_crew(request: AgentRequest) -> AgentResponse:
    try:
        llm = get_llm()
        summarizer = make_agent(
            llm, "Meeting Secretary",
            "Condense meeting transcripts into concise executive bullet points.",
            "You capture core discussion points without introducing unmentioned topics.",
        )
        action_finder = make_agent(
            llm, "Action Item Coordinator",
            "Extract every action item, responsible owner, and expected deadline.",
            "You format action items systematically, marking missing owners as 'Unassigned'.",
        )
        decision_finder = make_agent(
            llm, "Governance Recorder",
            "Document explicit agreements, resolutions, and unresolved blockers.",
            "You log confirmed agreements and distinguish them from open discussions.",
        )
        formatter = make_agent(
            llm, "Documentation Specialist",
            "Compile extracted details into a clean executive meeting memorandum.",
            "You organize disparate outputs into a professional Markdown structure.",
        )

        t1 = Task(
            description="Summarize this transcript in 5-7 focused bullets:\n{transcript}",
            expected_output="Executive summary bullet points.",
            agent=summarizer,
            async_execution=True,
        )
        t2 = Task(
            description="Extract all action items as 'Owner | Action | Due Date' from:\n{transcript}",
            expected_output="Bulleted list or table of actionable assignments.",
            agent=action_finder,
            async_execution=True,
        )
        t3 = Task(
            description="List all agreed decisions and open blockers from:\n{transcript}",
            expected_output="Numbered list of confirmed decisions and risks.",
            agent=decision_finder,
            async_execution=True,
        )
        final_task = Task(
            description="Assemble the summary, action items, and decisions into a finalized minutes document.",
            expected_output="Clean Markdown minutes with sections: Summary, Decisions, and Action Items.",
            agent=formatter,
            context=[t1, t2, t3],
        )

        output = run_crew([summarizer, action_finder, decision_finder, formatter], [t1, t2, t3, final_task], inputs={"transcript": request.input})
        return AgentResponse(output=save_artifact("meeting_notes.md", "Meeting Notes", output))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 3 error: {exc}")


# ---------------------------------------------------------------------------
# Project 4: Data Report Crew (Pandas Pre-Calculation + Review)
# ---------------------------------------------------------------------------
def compute_data_metrics(raw_text: str) -> str:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    if len(lines) < 2 or not any("," in l for l in lines[:3]):
        return "No valid CSV structure found. Perform a qualitative appraisal of the input."
    try:
        df = pd.read_csv(io.StringIO(raw_text))
        if df.empty:
            return "CSV parsed successfully but contains zero data rows."
        numeric = df.select_dtypes(include="number")
        stats = [
            f"Total Rows: {len(df)}",
            f"Columns: {', '.join(map(str, df.columns))}",
            f"Missing Values: {df.isna().sum().to_dict()}",
        ]
        if not numeric.empty:
            stats.append("Descriptive Statistics:\n" + numeric.describe().round(2).to_string())
        return "\n\n".join(stats)
    except Exception as e:
        return f"Unable to parse CSV ({e}). Analyze provided text carefully."


@app.post("/agent4", response_model=AgentResponse)
def data_report_crew(request: AgentRequest) -> AgentResponse:
    try:
        llm = get_llm()
        metrics = compute_data_metrics(request.input)
        analyst = make_agent(
            llm, "Quantitative Analyst",
            "Interpret data distributions, key trends, and statistical signals.",
            "You base conclusions strictly on calculated metrics and avoid hallucinating data.",
        )
        reviewer = make_agent(
            llm, "Audit & Quality Reviewer",
            "Verify analytical claims against data metrics and ensure balanced conclusions.",
            "You ensure findings distinguish causation from correlation and note data limitations.",
        )

        t_analysis = Task(
            description="Analyze the dataset and computed statistics:\n\nInput:\n{data}\n\nStatistics:\n{stats}",
            expected_output="Draft data analysis identifying patterns and metrics.",
            agent=analyst,
        )
        t_review = Task(
            description="Review the analysis. Organize into Executive Summary, Key Trends, Data Quality, and Recommendations.",
            expected_output="Polished Markdown data report with findings and verified figures.",
            agent=reviewer,
            context=[t_analysis],
        )

        output = run_crew([analyst, reviewer], [t_analysis, t_review], inputs={"data": request.input, "stats": metrics})
        return AgentResponse(output=save_artifact("data_report.md", "Data Report", output))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 4 error: {exc}")


# ---------------------------------------------------------------------------
# Project 5: Customer Support Flow (Classification & Priority Routing)
# ---------------------------------------------------------------------------
class CustomerSupportFlow(Flow[WorkflowState]):
    @start()
    def classify_request(self) -> str:
        text = self.state.user_input.lower()
        if any(w in text for w in ("refund", "charge", "invoice", "billing", "payment", "subscription")):
            self.state.category = "billing"
        elif any(w in text for w in ("bug", "crash", "error", "broken", "failed", "down", "issue")):
            self.state.category = "technical"
        elif any(w in text for w in ("cancel", "password", "login", "order", "shipping", "account")):
            self.state.category = "account"
        else:
            self.state.category = "general"
        return self.state.category

    @router(classify_request)
    def route_request(self):
        return self.state.category

    @listen("billing")
    def handle_billing(self) -> str:
        agent = make_agent(get_llm(), "Billing Support Specialist", "Resolve billing inquiries clearly.", "Empathetic and precise.")
        return ask_agent(agent, "Draft a support reply for this billing inquiry:\n{user_input}", "Polite billing reply.", {"user_input": self.state.user_input})

    @listen("technical")
    def handle_technical(self) -> str:
        agent = make_agent(get_llm(), "Technical Support Engineer", "Troubleshoot software defects.", "Structured troubleshooter.")
        return ask_agent(agent, "Draft a step-by-step diagnostic response for:\n{user_input}", "Diagnostic support reply.", {"user_input": self.state.user_input})

    @listen("account")
    def handle_account(self) -> str:
        agent = make_agent(get_llm(), "Account Support Representative", "Address account and order access requests.", "Security-conscious agent.")
        return ask_agent(agent, "Draft a customer support reply for:\n{user_input}", "Account guidance reply.", {"user_input": self.state.user_input})

    @listen("general")
    def handle_general(self) -> str:
        agent = make_agent(get_llm(), "General Support Representative", "Address general inquiries helpfully.", "Helpful concierge.")
        return ask_agent(agent, "Draft a helpful customer reply for:\n{user_input}", "Helpful support message.", {"user_input": self.state.user_input})

    @listen(or_(handle_billing, handle_technical, handle_account, handle_general))
    def finalize(self, response: str) -> str:
        self.state.output = f"**Category**: {self.state.category.upper()}\n\n{response}"
        return self.state.output


@app.post("/agent5", response_model=AgentResponse)
def customer_support_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(CustomerSupportFlow, request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 5 error: {exc}")


# ---------------------------------------------------------------------------
# Project 6: Resume Screening Flow (Structured Evaluation + Router)
# ---------------------------------------------------------------------------
class ResumeScreeningFlow(Flow[WorkflowState]):
    @start()
    def evaluate_fit(self) -> str:
        evaluator = make_agent(
            get_llm(), "Technical Talent Assessor",
            "Score candidate resume alignment against role criteria objectively.",
            "You evaluate skills and qualifications, ignoring non-job criteria.",
        )
        task = Task(
            description=(
                "Evaluate the candidate resume against role requirements:\n{user_input}\n\n"
                "Provide an objective score between 0 and 100 with clear rationale."
            ),
            expected_output="Structured score and rationale.",
            agent=evaluator,
            output_pydantic=ScoreResult,
        )
        crew = Crew(agents=[evaluator], tasks=[task], verbose=False)
        result = crew.kickoff(inputs={"user_input": self.state.user_input})
        
        parsed: ScoreResult = result.pydantic if hasattr(result, "pydantic") and result.pydantic else ScoreResult(score=60, rationale=str(result))
        self.state.analysis = f"Score: {parsed.score}/100\n\n**Rationale**:\n{parsed.rationale}"
        self.state.category = "shortlist" if parsed.score >= 70 else "reject"
        return self.state.category

    @router(evaluate_fit)
    def route_candidate(self):
        return self.state.category

    @listen("shortlist")
    def process_shortlist(self) -> str:
        self.state.output = f"### STATUS: SHORTLISTED\n\n{self.state.analysis}"
        return self.state.output

    @listen("reject")
    def process_rejection(self) -> str:
        self.state.output = f"### STATUS: NOT SHORTLISTED\n\n{self.state.analysis}"
        return self.state.output


@app.post("/agent6", response_model=AgentResponse)
def resume_screening_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(ResumeScreeningFlow, request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 6 error: {exc}")


# ---------------------------------------------------------------------------
# Project 7: Bug Triage Flow (Issue Router + Specialized Response)
# ---------------------------------------------------------------------------
class BugTriageFlow(Flow[WorkflowState]):
    @start()
    def categorize_ticket(self) -> str:
        text = self.state.user_input.lower()
        if any(w in text for w in ("bug", "crash", "error", "defect", "broken", "reproduce")):
            self.state.category = "bug"
        elif any(w in text for w in ("feature", "enhancement", "support", "add", "request")):
            self.state.category = "feature"
        else:
            self.state.category = "question"
        return self.state.category

    @router(categorize_ticket)
    def route_issue(self):
        return self.state.category

    @listen("bug")
    def triage_bug(self) -> str:
        agent = make_agent(get_llm(), "Defect Triage Engineer", "Isolate software defects and reproduction steps.", "Methodical QA lead.")
        return ask_agent(agent, "Triage this bug report. Identify severity, likely cause, and reproduction steps:\n{user_input}", "Structured bug triage note.", {"user_input": self.state.user_input})

    @listen("feature")
    def triage_feature(self) -> str:
        agent = make_agent(get_llm(), "Product Analyst", "Translate feature requests into agile user stories.", "User-centric PM.")
        return ask_agent(agent, "Draft a user story with acceptance criteria for:\n{user_input}", "Agile user story with acceptance criteria.", {"user_input": self.state.user_input})

    @listen("question")
    def triage_question(self) -> str:
        agent = make_agent(get_llm(), "Technical Support Specialist", "Answer technical questions helpfully.", "Knowledgeable engineer.")
        return ask_agent(agent, "Draft an answer or clarification request for:\n{user_input}", "Constructive technical guidance.", {"user_input": self.state.user_input})

    @listen(or_(triage_bug, triage_feature, triage_question))
    def finalize(self, response: str) -> str:
        self.state.output = f"**Issue Type**: {self.state.category.upper()}\n\n{response}"
        return self.state.output


@app.post("/agent7", response_model=AgentResponse)
def bug_triage_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(BugTriageFlow, request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 7 error: {exc}")


# ---------------------------------------------------------------------------
# Project 8: Content Repurposing Flow (Parallel Generation Joined with and_)
# ---------------------------------------------------------------------------
class ContentRepurposingFlow(Flow[WorkflowState]):
    @start()
    def extract_core_points(self) -> str:
        agent = make_agent(get_llm(), "Content Strategist", "Distill key themes and value points from source material.", "Editorial analyst.")
        self.state.analysis = ask_agent(
            agent, "Extract 4-5 core insights and narrative takeaways from this content:\n{user_input}",
            "Bulleted list of key content insights.", {"user_input": self.state.user_input}
        )
        return self.state.analysis

    @listen(extract_core_points)
    def generate_linkedin(self, insights: str) -> str:
        agent = make_agent(get_llm(), "LinkedIn Copywriter", "Craft engaging professional posts.", "Skilled social copywriter.")
        return ask_agent(agent, "Write a 150-200 word LinkedIn post based on these takeaways:\n{insights}", "Publish-ready LinkedIn post.", {"insights": insights})

    @listen(extract_core_points)
    def generate_twitter(self, insights: str) -> str:
        agent = make_agent(get_llm(), "X/Twitter Thread Specialist", "Craft concise viral threads.", "Punchy thread writer.")
        return ask_agent(agent, "Draft a 4-tweet thread (numbered 1/4 to 4/4) based on:\n{insights}", "Four-tweet thread sequence.", {"insights": insights})

    @listen(extract_core_points)
    def generate_newsletter(self, insights: str) -> str:
        agent = make_agent(get_llm(), "Newsletter Editor", "Draft readable newsletter segments.", "Engaging newsletter curator.")
        return ask_agent(agent, "Draft an executive newsletter section with a subject line and 3 takeaways based on:\n{insights}", "Markdown newsletter draft.", {"insights": insights})

    @listen(and_(generate_linkedin, generate_twitter, generate_newsletter))
    def compile_all(self, outputs: list[str]) -> str:
        linkedin_copy, twitter_copy, newsletter_copy = outputs
        self.state.output = (
            f"### 1. LinkedIn Post\n\n{linkedin_copy}\n\n---\n\n"
            f"### 2. Twitter / X Thread\n\n{twitter_copy}\n\n---\n\n"
            f"### 3. Newsletter Edition\n\n{newsletter_copy}"
        )
        return self.state.output


@app.post("/agent8", response_model=AgentResponse)
def content_repurposing_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(ContentRepurposingFlow, request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 8 error: {exc}")


# ---------------------------------------------------------------------------
# Project 9: Lead Qualification Flow (Structured Scoring & Tier Router)
# ---------------------------------------------------------------------------
class LeadQualificationFlow(Flow[WorkflowState]):
    @start()
    def qualify_prospect(self) -> str:
        agent = make_agent(
            get_llm(), "B2B Sales Qualification Analyst",
            "Evaluate sales lead profile against enterprise B2B ICP standards.",
            "Objective sales analyst evaluating authority, budget, and business fit.",
        )
        task = Task(
            description=(
                "Evaluate this sales lead against a B2B ICP profile (50+ employees or active tech usage):\n"
                "{user_input}\n\nAssign a qualification score from 0 to 100 with reasoning."
            ),
            expected_output="Structured score and evaluation.",
            agent=agent,
            output_pydantic=ScoreResult,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff(inputs={"user_input": self.state.user_input})
        
        parsed: ScoreResult = result.pydantic if hasattr(result, "pydantic") and result.pydantic else ScoreResult(score=50, rationale=str(result))
        self.state.analysis = f"Qualification Score: {parsed.score}/100\n\n**Rationale**:\n{parsed.rationale}"
        
        if parsed.score >= 75:
            self.state.category = "hot"
        elif parsed.score >= 45:
            self.state.category = "warm"
        else:
            self.state.category = "cold"
        return self.state.category

    @router(qualify_prospect)
    def route_tier(self):
        return self.state.category

    @listen("hot")
    def handle_hot(self) -> str:
        self.state.output = f"**TIER: HOT (High Priority)**\n\n{self.state.analysis}\n\n*Action*: Schedule direct outbound executive engagement."
        return self.state.output

    @listen("warm")
    def handle_warm(self) -> str:
        self.state.output = f"**TIER: WARM (Nurture Candidate)**\n\n{self.state.analysis}\n\n*Action*: Enroll in targeted case-study newsletter sequence."
        return self.state.output

    @listen("cold")
    def handle_cold(self) -> str:
        self.state.output = f"**TIER: COLD (Low Alignment)**\n\n{self.state.analysis}\n\n*Action*: Archive record in CRM; no immediate outreach needed."
        return self.state.output


@app.post("/agent9", response_model=AgentResponse)
def lead_qualification_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(output=run_flow(LeadQualificationFlow, request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 9 error: {exc}")


# ---------------------------------------------------------------------------
# Project 10: Approval Report Flow (Draft & Decision Router)
# ---------------------------------------------------------------------------
class ApprovalReportFlow(Flow[WorkflowState]):
    @start()
    def draft_report(self) -> str:
        agent = make_agent(get_llm(), "Executive Research Analyst", "Compile structured operational reports.", "Authoritative reporter.")
        self.state.analysis = ask_agent(
            agent, "Draft a report with Executive Summary, Findings, and Risks based on:\n{user_input}",
            "Draft report in Markdown.", {"user_input": self.state.user_input}
        )
        return self.state.analysis

    @router(draft_report)
    def evaluate_decision(self):
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
    def publish_approved(self) -> str:
        self.state.output = f"### STATUS: APPROVED & PUBLISHED\n\n{self.state.analysis}"
        return self.state.output

    @listen("revise")
    def apply_revision(self) -> str:
        editor = make_agent(get_llm(), "Report Editor", "Incorporate review feedback cleanly.", "Meticulous editor.")
        revised = ask_agent(
            editor, "Revise this report draft incorporating the user feedback:\n\nDraft:\n{draft}\n\nFeedback:\n{user_input}",
            "Revised Markdown report.", {"draft": self.state.analysis, "user_input": self.state.user_input}
        )
        self.state.output = f"### STATUS: REVISED DRAFT\n\n{revised}"
        return self.state.output

    @listen("pending")
    def request_review(self) -> str:
        self.state.output = (
            f"### STATUS: AWAITING APPROVAL\n\n{self.state.analysis}\n\n---\n"
            f"*Next Steps*: To simulate an approval cycle, submit again including either `'Approval: approve'` or `'Approval: revise: <feedback>'`."
        )
        return self.state.output


@app.post("/agent10", response_model=AgentResponse)
def approval_report_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        result = run_flow(ApprovalReportFlow, request.input)
        return AgentResponse(output=save_artifact("approval_report.md", "Approval Report", result))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 10 error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
