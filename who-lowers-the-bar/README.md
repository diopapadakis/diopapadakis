# Who Lowers the Bar? Expectation Management with a Bayesian Audience

This is a preliminary theory draft that grew out of the question: *does it make sense to lower an audience's expectations before a talk so that the talk comes as a pleasant surprise?*

| File | What it is |
|---|---|
| `who_lowers_the_bar.pdf` | The compiled draft (21 pages). |
| `who_lowers_the_bar.tex`, `references.bib` | LaTeX source. |
| `figures/` | Figures, produced by `code/figures.py`. |
| `code/model.py` | Numerical companion: closed forms, plus brute-force fixed points that check them. |
| `code/figures.py` | Regenerates every figure and every number quoted in the text. |

## The idea

A Bayesian audience cannot be talked into lower expectations. Its expected surprise is zero, and cheap talk about "how bad the paper is" gets ignored (Prop. 1). The audience does, however, judge a speaker relative to expectations for a purely Bayesian reason: it learns about her talent from a performance that also reflects her circumstances.

In the model, an agent of unknown talent can disclose verifiable evidence about her context (how hard her circumstances are) before a performance. An evaluator then hires her if his posterior clears a bar.

- **The pivotal-performance lemma (Lemma 2).** Disclosure changes how the performance is read, not how likely it is. So the agent discloses exactly when disclosure would raise her assessment at the performance that is pivotal under silence.
- **Why standing matters.** A Bayesian evaluator blames circumstances for disappointments and credits them for surprises. A favorite is pivotal after a disappointment and an underdog after a surprise. So favorites let the evaluator make their excuses for them, while underdogs pin the context down, and even *raise* the bar.

## Results and their status

| # | Statement | Status |
|---|---|---|
| Prop. 1 | Benchmarks. No systematic surprises. Cheap talk about context is equivalent to babbling. With linear payoffs, standing is irrelevant (plain Dye). | Proved |
| Lemma 2 | Pivotal performance: disclose iff disclosure raises the assessment at the silent hiring bar. | Proved; holds beyond the Gaussian model |
| Thm. 1 | Unique equilibrium, in closed form: disclose iff `d ≥ β(q)·σ_d·√(1+σ_d²/V) + (σ_d²/σ_θ²)·g`, where `β(q)` is the Dye constant. | Proved (Gaussian model plus a regularity condition) |
| Cor. 2 | For `g < g₀` (with `g₀ > 0`, small), the marginal discloser reveals a context *easier* than the evaluator would presume from silence, so she raises the bar. For `g > g₀`, favorites conceal contexts *harder* than presumed. | Proved |
| Cor. 3 | The evaluator learns strictly less about the agent the larger her lead, so reputations at the top are tested least. | Proved |
| Prop. 2 | Noise contexts. Favorites disclose "preliminary" (noisy) and underdogs disclose "polished" (precise). | Proved; no regularity needed; also checked numerically in a non-monotone case |
| Prop. 3 | Excuses made *after* the performance do not depend on the lead or on the shape of the payoff. Disclosure made *before* does. | Proved |
| Prop. 4 | Mandatory context disclosure helps underdogs and hurts favorites. | Proved for mandatory vs. no disclosure; the voluntary regime is shown numerically |
| Prop. 5 | With a hiring threshold, private information about one's own talent does not change disclosure, so nothing is signaled. | Proved |
| Obs. 1 | With smooth (S-shaped) stakes, the threshold is still increasing in the lead. | Numerical only |
| Conj. 1 | The same holds for general S-shaped stakes. | Conjecture |

## Closest literature and overlap check

- **Zwiebel (1995, JPE), "Corporate Conservatism and Relative Compensation".** This paper rules out my earlier "license to be modest" version, in which the agent has private ability and chooses a more or less benchmarkable project. That setup yields exactly the pattern where both extremes deviate and the middle conforms. The current draft avoids the overlap in two ways. The instrument is disclosure of the benchmark, which does not change the performance technology. As a result, nothing is signaled even when the agent has private information (Prop. 5).
- **Harbaugh & To (2020, JME), "False Modesty".** There, withholding good news is an equilibrium when prior expectations are favorable. The conclusion is similar to ours, but the mechanism differs: in their model the sender knows her type and the receiver has private information.
- **Dye (1985) and Jung & Kwon (1988).** Our equilibrium uses the same constant `β(q)`. In their model standing plays no role (Prop. 1(c)).
- **Kamenica & Gentzkow (2011); Hermalin (1993).** These give the "favorites prefer less information" logic. It does not pin down *which* contexts are disclosed.
- **Precision disclosure: Hughes & Pae (2004, JAE); Kim & Pae (2025, TAR).** These are related to Prop. 2. Their payoffs are linear in prices, disclosure is ex post, and they have no difficulty dimension.
- **Disclosure alongside exogenous information: Acharya, DeMarzo & Kremer (2011); Frenkel, Guttman & Kremer (2020); Libgober, Michaeli & Wiedman (2023 WP).** In these papers the disclosed and outside information concern the same value.
- **Expectations and reference points.**
  - Kőszegi–Rabin; Ely–Frankel–Kamenica (2015); Duraj–He (2024, TE).
  - Zeckhauser & Viscusi (2025, JRU), "Managed Expectations Theory": people lowering their *own* expectations.
  - Robertson (2025, JPET): judicial persuasion with reference dependence.
- **Self-handicapping.** Bénabou & Tirole (2002); Xiang, Gershman & Gerstenberg (*Cognition*): signaling to naive versus sophisticated observers.
- **Applications.**
  - Politics: Ashworth, Bueno de Mesquita & Friedenberg (2017); Garro (2019); Casey & Glennerster (2024) on front-runners avoiding debates.
  - Education: the Cornell median-grade experiment (Bar, Kadiyali & Zussman 2009); test-optional admissions (Dessein, Frankel & Kartik 2025).

**Remaining novelty risks, not fully checked.** Most publisher and arXiv full texts were blocked from this environment, so the comparisons above rely on abstracts and summaries. The places I would search next:

1. Accounting theory on disclosing the *context* of earnings (transitory items, attributions in the MD&A) under career concerns or meet-or-beat thresholds.
2. Formal political theory of pre-emptive blame-shifting by incumbents.
3. A 2025 arXiv paper, "Diagnostic Feedback under Hidden Task Difficulty". In it the *evaluator* knows the difficulty, and the focus is on feedback and motivation, so it looks different, but it should be read.

## Suggested next steps

1. Prove Conjecture 1 (S-shaped stakes), probably with a single-crossing argument on the marginal-payoff weights.
2. The commitment version: optimal institutional disclosure of context when the evaluator's information arrives exogenously.
3. Dynamics: whether favorites' hard-to-read performances make reputations persistent (a Matthew effect).
4. Empirics. Examples: seminar disclaimers by speaker seniority; firms pre-announcing adverse conditions when expected to miss versus beat; which side asks for "context" in grading and admissions.

## Rebuild

```bash
cd code && python3 figures.py        # needs numpy, scipy, matplotlib
cd .. && latexmk -pdf who_lowers_the_bar.tex
```
