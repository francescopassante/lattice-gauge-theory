# LGT learning resources

Curated list of textbooks, lecture notes, and ML-for-LGT papers, with a
one-line note on the main feature of each and a suggested reading order
for this thesis.

## Textbooks

- **Gattringer & Lang, *Quantum Chromodynamics on the Lattice* (2010)** — the modern standard graduate textbook; balanced theory + algorithms + analysis; start here.
- **Rothe, *Lattice Gauge Theories: An Introduction* (4th ed., 2012)** — most accessible intro for students; gentlest learning curve.
- **Montvay & Münster, *Quantum Fields on a Lattice* (1994)** — classic, formal/field-theoretic; the deep reference when Gattringer–Lang is too brief.
- **DeGrand & DeTar, *Lattice Methods for Quantum Chromodynamics* (2006)** — practical and algorithmic; the book to read for *how to actually simulate*.
- **Smit, *Introduction to Quantum Fields on a Lattice* (2002)** — concise and conceptual; strong on the physical picture, light on machinery.
- **Creutz, *Quarks, Gluons and Lattices* (1983)** — short, foundational, written by one of the inventors of lattice MC; still excellent for the basics of the Wilson action and Metropolis/heat-bath.
- **Knechtli, Günther & Peardon, *Lattice Quantum Chromodynamics: Practical Essentials* (2017, Springer Brief)** — ~140 pages, very modern and very practical; good "second book" companion to Gattringer–Lang.

## Lecture notes & reviews (free, often better than chapters of the books)

- **Kogut, *Introduction to Lattice Gauge Theory and Spin Systems* (Rev. Mod. Phys. 1979)** and *…QCD* (RMP 1983) — the canonical pedagogical reviews; still readable.
- **Lüscher's Les Houches / Nara lectures** — algorithms and chiral fermions by the master; terse but authoritative.
- **Sharpe's TASI lectures** — clean, pedagogical; good for fermions and chiral perturbation theory.
- **Wilson, *Confinement of Quarks* (Phys. Rev. D 10, 1974)** — the founding paper; surprisingly readable, worth doing once.

## ML-for-LGT entry points

- **Favoni et al. 2012.12901 (L-CNN)** — the architecture our G-GAT extends; already in our reading list.
- **Albergo, Kanwar, Shanahan 1904.12072** — the original flow-based sampling paper; short and clear, best first exposure to the idea.
- **Boyda et al. 2008.05456** — gauge-equivariant normalizing flows for SU(N); the reference flow paper.
- **Boyda et al., *Applications of Machine Learning to Lattice Quantum Field Theory* (2202.05838)** — Snowmass white paper; broad survey of the whole ML-for-LGT landscape.

## Suggested reading order for this thesis

1. **Rothe ch. 1–4** (or Gattringer–Lang ch. 1–4) — lattice action, links, plaquettes, Wilson loops. Build the basic mental model.
2. **DeGrand–DeTar ch. 1–4** — Metropolis / heat-bath / overrelaxation / HMC. This is what we'll implement in `sampling.md`.
3. **Gattringer–Lang ch. 4 + Creutz** — Cabibbo–Marinari specifically, for the SU(N) heat-bath kernel.
4. **Favoni 2012.12901 + Boyda 2008.05456** — once the architecture work begins (G-GAT block, eventually flow backbones in Phase 6).
