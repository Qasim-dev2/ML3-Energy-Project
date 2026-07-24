# Contributing Guide

Thank you for your interest in contributing to the **Smart Electricity Consumption Prediction** project!

## Getting Started

1. **Fork** this repository and clone your fork locally.
2. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `data/`   | Dataset files and generation script |
| `src/`    | ML pipeline scripts |
| `app/`    | Gradio web interface |
| `models/` | Saved model artifacts |
| `reports/`| Generated reports and plots |

## Code Style

- Follow **PEP 8** conventions.
- Add docstrings to all new functions and classes.
- Keep functions small and focused (single responsibility).
- Use the centralized `config.py` for paths and constants — avoid hardcoding.

## Submitting Changes

1. Ensure your changes don't break existing scripts by running the pipeline end-to-end.
2. Commit with clear, descriptive messages using [Conventional Commits](https://www.conventionalcommits.org/) style:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `chore:` for maintenance tasks
   - `refactor:` for code restructuring
3. Push your branch and open a **Pull Request** against `main`.

## Reporting Issues

Please open a GitHub Issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your Python version and OS
