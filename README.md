# Lesson Foundry

A Streamlit app that creates a beginner lesson, checks it against a hard pass/fail rubric, and retries with specific rejection reasons. It is designed for the assignment topic, **Introduction to RAG**, but accepts any topic.

## Run locally

Requires Python 3.11 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run main.py
```

Choose Google Gemini or OpenAI in the sidebar, then enter the matching API key. Keys can be entered in the password field for the current session or set as `GEMINI_API_KEY` / `OPENAI_API_KEY` in `.env`. The initial topic is Introduction to RAG. The default models are `gemini-3.8-flash` and `gpt-4o-mini`; change the model field if your account uses another model. Gemini requests use Google's Interactions API.

## How the system works

1. **Generate:** a teaching prompt requests a standalone lesson for a zero-background learner, using plain language, an explained example, and a recap.
2. **Evaluate:** a separate model call sees the topic, draft, and six binary checkpoints: accuracy and grounding, beginner language, worked example, explained jargon, complete coverage, and coherent flow. A checkpoint passes only when the evaluator returns an explicit boolean `true`.
3. **Regenerate:** any failed checkpoint sends its reason and evaluator suggestions into the next generation prompt. Choose one or two retries, for a strict maximum of two or three drafts.
4. **Remember:** SQLite stores drafts, evaluations, outcomes, and human feedback across app restarts. Repeated rubric failures and recent feedback become compact guidance for later generations.

The rejection log shows each draft, the failed checks, and whether their reasons were sent forward. A failed final draft is labeled **DID NOT PASS** rather than being presented as approved. Lessons can be downloaded as Markdown.

## Design choices and limits

- The generator and evaluator are separate calls with different instructions; using the same selected model keeps setup simple, but does not make the evaluator truly independent. For higher assurance, use a separately selected judge model or provider.
- The accuracy checkpoint is a model judgment, not external fact verification. The generator is told not to invent citations; important lessons still need human review and trusted sources.
- All checks are hard gates: no averaging hides a failure. Retries are bounded to control cost and prevent endless loops.
- SQLite was chosen over a vector database because the memory is structured run history and concise feedback, not semantic document retrieval. Configure another path with `LESSON_MEMORY_DB`.
- API keys are not written to the run database. For deployment, use Streamlit secrets or a managed secret store and protect the app with authentication.

## Tests

Run the provider-independent pipeline and persistence tests with:

```powershell
python -m unittest -v
```
