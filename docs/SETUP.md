# Setup Guide

This guide walks you through configuring every credential and service the platform needs.

---

## 1. Install the package

```bash
git clone <repo-url> experimenter
cd experimenter
pip install -e ".[dev]"   # full install with dev tools
# or: pip install -e "."  # production only
```

Copy the env template:
```bash
cp .env.example .env
```

---

## 2. Anthropic API key (required)

The planner and reporter both call Claude.

1. Go to https://console.anthropic.com/settings/keys
2. Create an API key
3. Set in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

---

## 3. Kaggle (optional — enables Kaggle dataset search)

1. Log in to https://www.kaggle.com
2. Go to **Account → API → Create New Token**
3. A `kaggle.json` file is downloaded with `username` and `key`
4. Set in `.env`:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

---

## 4. Weights & Biases (optional — enables experiment tracking)

1. Sign up / log in at https://wandb.ai
2. Go to **User Settings → API Keys → New key**
3. Set in `.env`:
   ```
   WANDB_API_KEY=your_key
   WANDB_PROJECT=experimenter   # or any project name you prefer
   ```

---

## 5. GCP / Vertex AI (optional — enables cloud training)

Skip this section if you only want local training.

### 5a. Create a GCP project

1. Go to https://console.cloud.google.com
2. Create a new project (or use an existing one)
3. Note the **Project ID** (not the display name)

### 5b. Enable APIs

```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

Or enable via the Cloud Console:
- Vertex AI API
- Cloud Storage API

### 5c. Create a GCS bucket

```bash
gcloud storage buckets create gs://my-project-experimenter-data \
  --location=us-central1
```

### 5d. Set up authentication

**Option A — Application Default Credentials (recommended for local dev):**
```bash
gcloud auth application-default login
```

**Option B — Service Account (recommended for CI/CD):**
1. Create a service account in IAM with roles:
   - `roles/aiplatform.user`
   - `roles/storage.objectAdmin`
2. Download the JSON key file
3. Set in `.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

### 5e. Set GCP values in `.env`

```
GCP_PROJECT_ID=my-project-id
GCP_REGION=us-central1
GCS_BUCKET=my-project-experimenter-data
```

---

## 6. Claude Desktop / Claude Code MCP integration

Add the server to your Claude MCP config.

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "experimenter": {
      "command": "experimenter-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "WANDB_API_KEY": "...",
        "GCP_PROJECT_ID": "...",
        "GCS_BUCKET": "...",
        "KAGGLE_USERNAME": "...",
        "KAGGLE_KEY": "..."
      }
    }
  }
}
```

**Claude Code** (add to `.claude/settings.json` in your project):

```json
{
  "mcpServers": {
    "experimenter": {
      "command": "experimenter-mcp"
    }
  }
}
```

The server reads credentials from `.env` in the working directory.

---

## 7. Verify setup

```bash
# Test the CLI (local mode, no cloud needed)
experimenter plan "Does exercise frequency predict academic performance?"

# Run a quick end-to-end smoke test (local compute, tiny dataset)
make smoke
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY not set` | Add key to `.env` |
| `Kaggle: 401 Unauthorized` | Check `KAGGLE_USERNAME` / `KAGGLE_KEY` |
| `W&B: API key invalid` | Run `wandb login` or set `WANDB_API_KEY` |
| `GCP: Permission denied` | Run `gcloud auth application-default login` |
| `No module named experimenter` | Run `pip install -e .` from repo root |
