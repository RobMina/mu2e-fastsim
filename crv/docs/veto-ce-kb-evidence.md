# CE normalization: authenticated KB evidence

## Status

The documented on-host KB client authenticates successfully. The earlier HTTP401
in [veto-ce-reference-evidence.md](veto-ce-reference-evidence.md) came from an
unauthenticated request; it is **not a continuing access blocker**. The original
capture remains unchanged. This follow-up uses the existing service credential
in-process on its host, without displaying/copying the key or putting it in a
synced configuration. The service URL was checked against the live registry.

This is documentary discovery only: no fit, physics extraction, model application,
new response draw or redefinition of a viewed sample as holdout. No independent
measured CE calibration population has been admitted. The historical F values,
C4 stock choice and unestimated finite-library uncertainty remain unchanged.

## Search and evidence handling

Direct `kb_search` and `kb_get` only; no `kb_research` or other synthesized answers.
Searches request at most four results, restricted to Mu2e DocDB. Some subsequent
searches additionally restrict to text and disable reranking. Search-result
summaries are discovery aids, **not primary evidence**: the relevant documents
below were retrieved with `kb_get` and conclusions checked against the returned
full text and its page markers. Embedded figure descriptions are not visual
inspection, and no fit coefficient or plot precision was independently extracted.
The stored text is a KB extraction, not a newly downloaded/checksummed original
PDF; extracted equations may be garbled. No proposed normalization equation is
implemented from that extraction.

One initial long hybrid query, `IHEP proton bunch intensity measurements lognormal
spill duty factor`, hit the 120-second client timeout, exit124, with zero output.
It is **not a zero-result search**. The initial wrapper's `set -e` plus AND-list
status check mistakenly continued and printed a completion line; the saved
pipeline status `124 0` is authoritative. The failure was preserved, the query
was not repeated, and client processes were confirmed absent. The wrapper now
uses an explicit conditional exit on either failed pipeline component. A client
timeout does not certify cancellation of a server-side search.

A distinct fulltext `IHEP` query returned unrelated antiproton/COMET leads in
its top three results; these were not treated as evidence against the relevant
measurement's existence. A narrower conceptual PBI search then found the
simulation review below. The exact-number `35621` search returned documents that
cite it, not the source document itself. No coverage/completeness claim is made.

Completed: authenticated tools/list, six successful searches and eight full-text
gets; the initial timed-out search is separate. Per-call captures are capped at
1MiB stdout, 256MiB client address space and120s with no retry; no response
exceeded the cap. Shell syntax checks pass. Successful responses have status
`0 0`; search responses are checked as JSON and retrieved documents carry an
end-of-document marker. These checks validate capture integrity, not scientific
calibration or independence.

## Retrieved references and their scope

| Reference | Directly relevant full-text evidence | What it does not establish |
|---|---|---|
| [53310 v1: Luminosity Stream Analysis](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=53310), Hope Applegate, 8 July2025 | Page5 names `dig.mu2e.NoPrimaryMix1BBTriggerable.MDC2020ae_best_v1_3.art` as pure background. Page14 lists prediction checks and inter-observable correlations as future work. | Not independently measured beam data, not a CE-containing validation population, and no demonstrated ancestry separation from the retained libraries. The dataset name is a new documentary lead; it was not queried or opened. |
| [53761 v1: Luminosity stream status](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=53761), Michael MacKenzie/Giani Pezzullo, 25 August2025 | Page2 proposes storing quantities independently of trigger decision. Pages4–7 discuss 2BB observables, DeltaFinder proton clusters and calo energy; page7 explicitly says sensitivity to systematics/calibrations is unclear. Page18 reports about8.5% event-level **statistical** precision for calo-energy POT estimation. | No validated8.5% total uncertainty for the retained1BB CE sample. No coefficient, calibration uncertainty or target weight is imported. Trigger-independent design intent is not proof of the inclusion law of a named retained output. |
| [44236 v1: RMC as a tool for the stopped muon rate normalization](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=44236), Pasha Murat, 22 December2022 | Pages2–3 distinguish integrated STM absolute normalization from fast proxies; tracker hits track beam intensity rather than directly the stop rate. Pages5–8 propose proton counting plus STM/RMC normalization. | The quoted expected statistical/systematic precisions are proposal/context, not an executed CE calibration or uncertainty on F. |
| [56602 v1: Tracker contribution to normalization](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=56602), Vadim Rusu, 5 May2026 | Page2 labels this a summary of earlier work/private discussions. Page5 cites Edmonds31700 for PBI versus straw-digi count; page6 cites Brown/Bonventre35621 for early-hit timing and says commissioning learning is needed. Page8 warns of model dependence. | Neither cited underlying study was fetched here; a secondary review is not a deployed timing calibration/IOV. |
| [45595 v2: Computing and Software Review: Algorithms](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=45595), Dave Brown, revised17 May2023 | Page12 explicitly describes a lognormal fit of IHEP slow-extraction data, scalable absolute intensity and conservative startup parameters, alongside Werkema's13 April2020 spill-simulation slide. | Connects the existing source-macro lead to collaboration documentation, but does not identify the original IHEP measurement/run/calibration/uncertainties or establish the selected CE joint law. |
| [43581 v2: Data Flow and Processing of Intensity Related Information](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=43581), Robert Kutschke, 20 October2022 | Pages9–11 describe PBI-dependent reconstruction efficiency and fast intensity proxies; page20 proposes combining relative time structure with integrated MSTM normalization. | A data-flow/analysis design, not evidence that the proposed database or event association is deployed for the retained sample. |
| [34184 v7: On Mu2e Normalization](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=34184), James Miller, revised6 November2023 | Pages2–3 require pulse history and rate/time-dependent efficiencies; random pulse sampling does not preserve pulse-to-pulse correlations. Page7 proposes minimum-bias CRV accidental-veto-rate characterization, distinct from cosmic rejection inefficiency. Page14 warns about using the same detectors for selection and normalization and suggests independent monitors/minimum-bias data. | Explicitly a proposed strategy with open accuracy/implementation questions, not a validated correction formula. Carry-over percentages in different slides are not adopted as constants. |
| [44084 v4: MDC2020 Simulation Campaign](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=44084), revised26 February2023 | PBI-simulation section distinguishes a lognormal model from a dedicated simulated spill sequence; cites Werkema32620 and33344. Pileup section documents fixed bunch-time/lifetime samplings and concerns about oversampling artifacts, especially neutrals/electron backgrounds and accidental coincidences. | MDC2020 source-model history, not proof of MDC2025 worker deployment, independent physical histories or covariance for the retained library. |

Exact identifiers used for the full-text gets:

```text
53310: 4d14a39c-099f-4767-8482-c05d1f88f5c6
53761: 054dfb21-2e12-4f46-b9f2-a511c4814775
44236: 158384ef-3264-48af-9e56-d324e16627b3
56602: 7aba4e50-9bc7-44d0-943a-1dc4f910adbd
45595: 8d782f35-afe6-47e1-bf65-88ed5f769a09
43581: 740902d6-d04b-4a0b-9211-65f8d0fbdbea
34184: a936248d-c256-4168-8dab-0c692ae958b0
44084: c77630c7-c5b4-4412-9268-8485ec5c1e7d
```

## Consequence for the next step

There is now a concrete collaboration provenance trail, not just local source
names. Three evidence paths can be pursued without fitting existing rows:

1. Follow **32620/33344** for the proton-pulse/explicit-spill model and the original
   IHEP measurement citation. Keep measured spectra separate from simulated
   sequences and conservative model parameters.
2. Follow **31700/35621** and contact the luminosity-study owners for an executed
   calibration sample, conditions and timing reference. The pure-background
   dataset named by53310 is a possible simulation cross-check, **not** an
   automatically independent validation sample. Its ancestry/inclusion law and
   transport to1BB CE-containing events would need explicit admission.
3. Use34184/43581 to specify the required **absolute normalization plus relative
   pulse-history/time information**, independently measured where possible, and
   the bridge to selected-track F. Minimum-bias inclusion probabilities, shared
   detector effects, selection/time conditioning and background carry-over must
   be retained. No scalar calo-energy or proton-count regression alone closes it.

The calibration/validation and finite-library requirements remain those in
[veto-ce-calibration-validation.md](veto-ce-calibration-validation.md). Documents
and new source leads do not supply a holdout, a target law, a calibrated
uncertainty, or permission to recompute F. Existing successful products remain
frozen and reusable.
