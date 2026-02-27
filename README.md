<a name="readme-top"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/All-Hands-AI/docs/main/openhands/static/img/logo.png" alt="Logo" width="200">
  <h1 align="center">OpenHands: Code Less, Make More</h1>
</div>


<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://all-hands.dev/joinslack"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

  <hr>
</div>

Welcome to OpenHands (formerly OpenDevin), a platform for software development agents powered by AI.

OpenHands agents can do anything a human developer can: modify code, run commands, browse the web,
call APIs, and yes—even copy code snippets from StackOverflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.


> [!IMPORTANT]
> **Upcoming change**: We are renaming our GitHub Org from `All-Hands-AI` to `OpenHands` on October 20th, 2025.
> Check the [tracking issue](https://github.com/All-Hands-AI/OpenHands/issues/11376) for more information.


> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

## ☁️ OpenHands Cloud
The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev),
which comes with $20 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The easiest way to run OpenHands locally is using the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation from your current project's virtual environment and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
source .venv/bin/activate
make run                    # Both frontend and backend
# OR
make start-backend          # Backend only (port 3000)
make start-frontend         # Frontend only (port 3001)
```

**Update Python dependencies**:

```bash
source .venv/bin/activate
uv pip install --upgrade package-name    # Update specific package
uv pip install --upgrade -e .            # Update all packages
```

Open [http://localhost:3000](http://localhost:3000) to use OpenHands.

## Getting Started

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4.5](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-5-20250929`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).
