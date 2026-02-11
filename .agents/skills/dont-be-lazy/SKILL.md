---
name: dont-be-lazy
description: Thorough debugging and root cause analysis for traceback errors and runtime breaking implementation flaws. Use when encountering errors, exceptions, crashes, test failures, or broken functionality to ensure deep investigation and complete fixes rather than surface-level patches.
---

# Don't Be Lazy - Thorough Debugging

When errors occur, do the complete work necessary to identify and fix the root cause. Never apply surface-level patches or guess at solutions.

## Core Principle

**Understand before fixing.** If you don't fully understand why an error is happening, you cannot fix it properly. Do the investigation work.

## Error Response Workflow

When presented with a traceback, error message, or broken functionality:

### Step 1: Acknowledge What You See
State clearly:
- What error occurred
- Where it occurred (file, line, function)
- What the immediate symptom is

### Step 2: Investigate the Root Cause

Do NOT guess. Investigate:

1. **Read the traceback carefully**
   - Find the actual error line
   - Trace the call stack
   - Identify the triggering input/state

2. **Examine the relevant code**
   - Read the function where error occurred
   - Read calling functions
   - Check recent changes if applicable

3. **Understand the data flow**
   - What input caused this?
   - What state led here?
   - What assumptions were violated?

4. **Check context**
   - Related test files
   - Similar implementations elsewhere
   - Configuration that might affect behavior

### Step 3: Explain the Root Cause

Before proposing any fix, explain to the user:
- **Why** the error is happening (root cause)
- **What** assumption or condition was violated
- **Where** the breakdown occurred in the logic/data flow

### Step 4: Verify Understanding (if needed)

If uncertain about the root cause:
- Ask clarifying questions
- Add debug logging/output to confirm hypothesis
- Run tests to isolate the issue

### Step 5: Implement the Fix

Only after understanding the root cause:
- Fix the actual problem (not the symptom)
- Ensure the fix addresses the root cause
- Consider edge cases the fix might introduce
- Run tests to verify the fix works

## What "Don't Be Lazy" Means

### ❌ Lazy (Don't Do This)
- "Probably a syntax error, try fixing line X"
- "Add a try/except here to catch it"
- "Just add this import and see if it works"
- Changing code without reading it first
- Guessing based on error message only

### ✅ Thorough (Do This)
- Trace the full execution path
- Read all related code
- Understand data flow and state
- Explain the root cause before fixing
- Verify fix with tests
- Check for similar issues elsewhere

## Debugging Checklist

Before claiming an issue is fixed, verify:

- [ ] You can explain the root cause
- [ ] The fix addresses the root cause, not just symptoms
- [ ] Tests pass with the fix
- [ ] No new issues were introduced
- [ ] Similar code patterns checked for same issue

## Guiding the User Through Investigation

When debugging together:

1. **Show your work** - Explain what you're checking and why
2. **Narrow systematically** - Use binary search approach to isolate the issue
3. **State hypotheses** - "I think X is happening because Y. Let me verify by checking Z"
4. **Confirm before fixing** - "The root cause is X. Here's how I'll fix it: ..."
5. **Verify after fixing** - "Fix applied. Running tests to confirm..."

## Common Anti-Patterns to Avoid

| Anti-Pattern | Why It's Wrong | What To Do Instead |
|--------------|----------------|-------------------|
| Band-aid fixes (try/except to hide errors) | Hides bugs, doesn't fix them | Fix the actual cause |
| Shotgun debugging (change many things at once) | Can't tell what fixed it | Change one thing, test, repeat |
| Assuming without verifying | Often wrong, wastes time | Check the code/data to confirm |
| Stopping at first symptom | Root cause may be deeper | Trace full execution path |
| "It works on my machine" | Environment issues are real bugs | Understand environment differences |

## When You're Stuck

If you've investigated thoroughly and still don't understand:

1. **Summarize findings** - "Here's what I know..."
2. **State what's unclear** - "I don't understand why X happens"
3. **Request specific information** - "Can you show me the output of Y?"
4. **Propose next diagnostic step** - "Let me add logging to see Z"

Never guess and hope. Always know before fixing.
