# cyber-security_project# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and suggests concrete improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for targeted keyword matching
- **ATS score (0-100)** with a breakdown: keyword match, formatting, content quality, impact and metrics, readability
- Prioritised **improvements**, **missing keywords**, **formatting issues**, **before/after bullet rewrites** and strengths
- Downloadable Markdown report

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey. If you do not set the variable, the app asks for the key in the sidebar.

## Configuration

| Setting | How | Default |
|---|---|---|
| API key | `GEMINI_API_KEY` env var, Streamlit secret, or sidebar box | none |
| Model | `GEMINI_MODEL` env var or sidebar box | `gemini-3.5-flash` |

If the chosen model name is not found, the app automatically retries with `gemini-2.5-flash`.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

> **Note:** a key stored in secrets is used by every visitor of your public app, so you pay for (or use up the quota of) all their requests. Make the app private, or remove the secret so visitors enter their own key in the sidebar.

## Privacy

Resume text is sent to the Gemini API for analysis. The app itself stores nothing. Never commit API keys to GitHub.

## Limitations

- Scores are AI estimates, not the output of any real ATS.
- Scanned or image-only PDFs contain no text; use a text-based PDF or DOCX.
- Very long resumes are truncated to about 30,000 characters.
