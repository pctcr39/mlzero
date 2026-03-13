# /next-phase

Complete the current phase and advance to the next one.

## Steps

1. Read CLAUDE.md to know current phase
2. Read ML_Learning_Guide.md Progress Tracker
3. Verify all checkboxes for current phase are done:
   - If any unchecked → list what remains, don't advance
   - If all checked → proceed

4. Run a quick review:
   - Call `code-reviewer` on current phase's main file
   - Fix any remaining issues

5. Update CLAUDE.md:
   - Mark current phase complete
   - Set new "Current Learning Phase"
   - Update "Active file"

6. Update ML_Learning_Guide.md:
   - Mark all current phase items ✅
   - Add completion date

7. Call `ml-tutor` to introduce the NEXT phase:
   - What is it?
   - Why does it come after the current phase?
   - What will we build?
   - What new math concepts appear?

8. Create the next phase's first implementation file using `/learn-algorithm`

## Phase Progression
```
Phase 1: NumPy ✅
Phase 2: Linear Regression → Polynomial → Ridge/Lasso
Phase 3: Logistic Regression → Decision Tree → Random Forest → SVM → KNN
Phase 4: K-Means → DBSCAN → PCA
Phase 5: Self-Training → Label Propagation
Phase 6: Q-Learning → Policy Gradient
Phase 7: Single Neuron → MLP → Backprop
Phase 8: CNN → RNN/LSTM → Transformer
Phase 9: Fine-tuning → RAG → Agents
```
