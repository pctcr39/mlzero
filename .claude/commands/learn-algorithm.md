# /learn-algorithm

Walk through a full algorithm learning cycle from theory to implementation.

## Input: $ARGUMENTS (algorithm name, e.g. "Logistic Regression")

## Phase 1: Theory (ml-tutor agent)
1. Call `ml-tutor` to explain:
   - What problem does this algorithm solve?
   - What is the intuition / real-world analogy?
   - The mathematical formula with every symbol defined
   - Where it fits in the ML pipeline
   - When to use it vs alternatives

## Phase 2: Math (mathematician agent)
2. Call `mathematician` to:
   - Derive the gradient/update formula from scratch
   - Show the chain rule steps
   - Map final formula to NumPy code

## Phase 3: Implementation
3. Create a new file: `[phase_folder]/[category]/[algorithm]_scratch.py`
   - Top docstring: THEORY, MATH, VARIABLES
   - Class inheriting from core.base.BaseModel
   - fit(), predict(), score() methods
   - All variables explained (local/instance)
   - Demo code + exercises in `if __name__ == "__main__":`

## Phase 4: Verification (sklearn version)
4. Create `[algorithm]_sklearn.py`:
   - Import sklearn equivalent
   - Use same dataset
   - Compare: scratch results vs sklearn results
   - Print both side by side

## Phase 5: Experiments
5. Call `experimenter` to:
   - Design 3-5 experiments to build intuition
   - Run each and document findings

## Phase 6: Update Progress
6. Check off completed items in ML_Learning_Guide.md Progress Tracker
7. Add key patterns to CLAUDE.md "Patterns Learned So Far"
8. Update CLAUDE.md "Current Learning Phase"
