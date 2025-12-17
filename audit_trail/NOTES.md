Here are the Git commands for your Arch Linux terminal to save your Copilot-Opus work and start fresh:

### 1. Information Gathering (Optional but Recommended)
```bash
git status                    # See your uncommitted changes
git log --oneline -3          # View recent commits to confirm your starting point
git branch                    # See all existing branches
```

### 2. Save Current Work to 'caching-copilot-opus' Branch
```bash
git checkout -b caching-copilot-opus
git add .                     # Stage all changes (or specify files: git add file1 file2)
git commit -m "feat: prompt caching implementation using Copilot-Opus"
```

This creates a new branch containing all your current uncommitted work and commits it.

### 3. Return to Original State & Create New Branch
```bash
git checkout prompt-caching   # Go back to your original branch (clean working directory)
git checkout -b caching-opus-api
```

You now have a fresh branch starting from the exact same commit where you began, ready for your Anthropic API attempt.

### 4. Verify & Start Working
```bash
git status                    # Should show clean working directory
git branch                    # Should show you're now on 'caching-opus-api'
```

---

### Later: Comparing the Branches
When you're ready to compare the two attempts:
```bash
git diff caching-copilot-opus..caching-opus-api     # Show code differences
git log --oneline -10 --graph --all                 # Show commit history of both branches
git merge-base caching-copilot-opus caching-opus-api # Find the common ancestor commit
```

**Note**: If you had already pushed `prompt-caching` to remote and want to keep the branch structure clean, you could optionally delete the `prompt-caching` branch locally since it now serves as your "base" rather than active development branch.

```
   audit-trail/
   ├─ 2025-12-17_001-prompt.md        # your prompt
   ├─ 2025-12-17_001-copilot.md      # raw Copilot chat log
   ├─ 2025-12-17_001-diff.patch      # diff that resulted
   ├─ 2025-12-17_001-notes.md        # any extra notes
```

Create the patch while you are on the code branch (the one that contains the real changes), then copy it into the orphan `audit-trail` branch.

1. Stage everything you want to keep in the patch:

   ```bash
   git add llm_anthropic.py pyproject.toml …
   ```

2. Generate the patch (staged vs. last commit):

   ```bash
   git diff --cached > ../2025-06-11_001-diff.patch
   ```

   or, if you already committed, just diff that single commit:

   ```bash
   git show --pretty=email <commit> > ../2025-06-11_001-diff.patch
   ```

3. Switch to the audit branch and drop the file in:

   ```bash
   git checkout audit-trail
   mkdir -p audit-trail
   mv ../2025-06-11_001-diff.patch audit-trail/
   git add audit-trail/2025-06-11_001-diff.patch
   git commit -m "session 001: add resulting diff"
   git push
   ```

Repeat for every session; each patch lives only in the orphan branch, leaving the upstream-bound history untouched.