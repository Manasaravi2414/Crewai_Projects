CrewAI Projects API — 10 projects
What is included
Endpoint	Project	Pattern
POST /agent1	Blog Post Crew	Sequential Crew
POST /agent2	Competitor Analysis Crew	Hierarchical Crew
POST /agent3	Meeting Notes Crew	Parallel analysis tasks + formatter
POST /agent4	Data Report Crew	pandas summary + analysis/review Crew
POST /agent5	Customer Support Flow	Request category routing
POST /agent6	Resume Screening Flow	Score and route for shortlist/review
POST /agent7	Bug Triage Flow	Bug / feature / question routing
POST /agent8	Content Repurposing Flow	Parallel content variants
POST /agent9	Lead Qualification Flow	Hot / warm / cold routing
POST /agent10	Approval Report Flow	Draft / approve / revise workflow


Windows setup
1. Install Python and open this folder in VS Code.
2. Create a virtual environment:
   py -m venv .venv
   .venv\Scripts\Activate.ps1
3. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Create a .env file in this folder:
   GEMINI_API_KEY=put_your_gemini_api_key_here
   GEMINI_MODEL=gemini/gemini-3.5-flash-lite
   If that model is not available to your account, set GEMINI_MODEL to a Gemini model supported by your CrewAI/LLM setup.
5. Start the API:
   uvicorn app:app --reload
6. Open http://127.0.0.1:8000/docs.
Postman test
- Method: POST
- URL: http://127.0.0.1:8000/agent1 (change the endpoint to test each project)
- Body → raw → JSON:
  {
    "input": "Write a blog post about how AI helps students learn."
  }
Every endpoint responds with {"output": "..."}. The API uses the same request format for all projects.
Suggested test inputs
- /agent1: Artificial intelligence in education
- /agent2: Compare Notion, Trello, and Asana for small student teams. Verify all current pricing before making a purchasing decision.
- /agent3: Paste meeting transcript or rough meeting notes, including names, decisions, owners, and deadlines.
- /agent4: Paste a small CSV table, including the header row and several records.
- /agent5: I was charged twice for my subscription and need a refund.
- /agent6: Include a job description and resume text. Treat output as decision support only; a human must make hiring decisions.
- /agent7: Bug: The app crashes with an error when I click Save.
- /agent8: Paste a blog paragraph or article to repurpose.
- /agent9: Provide lead/company context, role, need, budget, and timeline if known.
- /agent10: Include report material. For a first run, no approval marker means pending human approval. To simulate feedback, include Approval: approve or Approval: revise in the input.
Notes and limitations
- The program is syntax-checked, but it has not been run end-to-end against your Gemini account. First run may reveal version/model configuration differences.
- CrewAI Flow APIs can differ between versions. If you receive an import or routing error, check the installed CrewAI version and adjust accordingly.
- Project 2 does not scrape websites automatically; provide verified competitor details in the input if the report requires current facts.
- Project 4 computes numeric summaries when the input is CSV text. For best results, paste valid CSV with a header row.
- Project 6 is an assistive scoring prototype, not an automated hiring decision system.
- Project 10 simulates approval within a stateless REST request. A production approval workflow should persist the draft and reviewer decision between requests.
- Markdown artifacts are saved under the outputs/ directory.
