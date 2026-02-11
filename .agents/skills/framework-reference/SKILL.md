---
name: framework-reference
description: Workflow for working with open-source frameworks by prioritizing reference material from the project directory. Use when modifying, extending, or configuring code that uses an open-source framework (React, Django, LangGraph, etc.) to ensure changes align with the framework's official patterns and conventions found in the project.
---

# Framework Reference Workflow

Enforce a reference-first approach when working with open-source frameworks to ensure changes align with official patterns.

## Core Workflow

When asked to make changes involving an open-source framework:

### Step 1: Identify the Framework
Determine which open-source framework(s) are involved in the request.

### Step 2: Search for Reference Material
Before making any changes, search the project directory for:

1. **Official documentation files** (`README.md`, `*.md` in docs/)
2. **Example files** (`examples/`, `samples/`, `demo/`)
3. **Configuration templates** (`*.example.*`, `*.template.*`)
4. **Test files** that demonstrate usage patterns (`tests/`, `*_test.py`, `*.spec.*`)
5. **Type definitions** or interfaces (`.d.ts`, protocol files)

### Step 3: Use Reference Material

If reference material is found:
- Read relevant reference files
- Follow the patterns and conventions shown
- Use the same structure, naming, and configuration

If reference material is NOT found:
- Proceed with general knowledge
- Note that no project-specific reference was available

### Step 4: Make Changes
Apply changes following the patterns from reference material, or using best practices if none found.

## Search Strategy

Use these patterns to find reference material:

```bash
# Documentation
glob "**/README*"
glob "**/docs/**/*.md"
glob "**/CONTRIBUTING*"

# Examples
glob "**/examples/**/*"
glob "**/samples/**/*"
glob "**/demo/**/*"

# Tests (show usage patterns)
glob "**/tests/**/*"
glob "**/*_test.*"
glob "**/*.spec.*"

# Configuration examples
glob "**/*.example.*"
glob "**/*.template.*"
glob "**/config/**/*"

# Type definitions
glob "**/*.d.ts"
glob "**/types/**/*"
```

## Example Workflow

User: "Help me add a new agent to this LangGraph orchestrator"

1. **Identify**: This involves LangGraph framework
2. **Search**: Look for existing agent implementations in the project
3. **Reference**: Read existing agent files to understand the pattern
4. **Apply**: Create new agent following the same structure

## Decision Tree

```
Is an open-source framework involved?
├── No → Proceed with general knowledge
└── Yes → Search for reference material
    ├── Found → Read and follow the patterns
    └── Not found → Proceed, note lack of reference
```

## Important Notes

- **Always search first** - Even if you know the framework well, project-specific conventions may differ
- **Prefer project examples over general knowledge** - Projects often have custom patterns
- **Look at tests** - Tests often show the cleanest usage examples
- **Check multiple examples** - If multiple patterns exist, choose the most recent or most used
