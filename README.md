# careloop_noshow_guard
End-to-end appointment no-show analysis for CareLoop Clinics: data cleaning in Python, SQL reporting, no-show risk prediction, and Gemini-assisted follow-up note review.

# CareLoop Clinics — No-Show Guard

## Summary
- **Part 1 (Data Cleaning):** Cleaned 1,020 raw bookings down to 990 valid appointments (218 no-shows, 22.0%), capped 26 distance outliers and 3 fee entry errors, and enriched all 8 clinics via Open-Meteo.
- **Part 2 (SQL Analysis):** Built `noshowguard` and found no-show rates rise sharply with booking lead time, jumping from 16.2% (1–7 days) to 30.0% (15–30 days) and 60.5% (31+ days).
- **Part 3 (Machine Learning):** Selected Logistic Regression over Decision Tree and Random Forest via 5-fold CV (ROC-AUC ~0.77 CV, ~0.79 test) using a 0.35 probability threshold (test recall ~0.57, ~49 flagged).
- **Part 4 (GenAI Follow-up):** Structured the first 10 call-centre notes with Gemini Flash and Pydantic validation, confirming that missed reminders (`forgot_or_no_reminder`) align with `reminder_sent = No`.

## Recommendation
CareLoop Clinics should prioritise automated reminders and evening-before confirmation calls for bookings made more than 14 days ahead, because no-show rates climb from 16.2% within a week to 30.0% at 15–30 days and 60.5% past 30 days, while appointments with reminders show roughly 69% lower odds of a no-show in the fitted model. Using the 0.35 risk threshold flags roughly a quarter of next-day visits (~49 of 198 test bookings) and catches more than half of actual no-shows (~57% recall), giving front-desk teams at high-rate clinics like Hadapsar Care Centre (27.3%) and Malviya Nagar Clinic (26.8%) a focused call list before slots go empty.

## Model Choice
I chose **Logistic Regression** (wrapped in a `StandardScaler` pipeline) because it achieved the highest 5-fold cross-validated ROC-AUC (~0.77 vs ~0.68 for the depth-3 tree and ~0.75–0.77 for Random Forest) and generalised cleanly to the test set (ROC-AUC ~0.79) without overfitting. I chose a **0.35 threshold** from out-of-fold training probabilities because it raised F1 from ~0.35 to ~0.49 and nearly doubled recall (~0.25 to ~0.47), which fits clinic operations where an empty doctor slot costs much more than a one-minute phone call. *(Update exact decimals from your `03_model.ipynb` run if needed.)*

## Limitations
1. **Small follow-up sample:** Only 10 call-centre notes were analysed due to free-tier API limits, so the root-cause breakdown is illustrative rather than representative of all clinics.
2. **Pre-split cleaning leakage:** Medians, quartiles, and outlier fences in Part 1 were computed across all 990 rows before the train/test split; in production these statistics would be fit on training data only.
3. **Random split and repeat patients:** Because the 80/20 split is random rather than chronological—and the same `patient_id` can appear in both train and test—real-world future performance may be slightly lower, and the data are observational so associations do not prove causation.
