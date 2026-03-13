# /review-code

Review a ML implementation for correctness and learning quality.

## Input: $ARGUMENTS (file path, e.g. "01_supervised/regression/linear_scratch.py")

## Steps

1. Read the file at $ARGUMENTS
2. Read CLAUDE.md to understand current learning phase
3. Call `code-reviewer` agent to:
   - Check math correctness against theory
   - Verify variable explanations are clear
   - Check gradient direction and update rule
   - Verify train/test split exists
   - Check for exercises at the bottom
4. Ask `mathematician` agent to verify:
   - Gradient formula matches derivation
   - Matrix shapes are correct
5. Generate review report with:
   - What's correct ✅
   - What needs fixing 🐛
   - Understanding questions to test comprehension 💡
   - Suggested next experiments 🎯
6. If critical bugs found → call `debugger` agent to explain and fix
7. Update CLAUDE.md with any new patterns learned
