const pptxgen = require("pptxgenjs");
const fs = require("fs");
const A = JSON.parse(fs.readFileSync("/home/claude/pk/BAN6800_MOD5_Package/artifacts/analysis.json"));
const MIT = JSON.parse(fs.readFileSync("/home/claude/pk/BAN6800_MOD5_Package/artifacts/mitigation.json"));
const R = JSON.parse(fs.readFileSync("/home/claude/pk/BAN6800_MOD5_Package/artifacts/model_results.json"));
const URL = process.env.DASH_URL || "https://YOUR-APP-NAME.streamlit.app";

const OP = A.operating.xgboost.at_recall_target, OP5 = A.operating.xgboost.at_default_0_5;
const F = A.fairness, D = A.descriptive, CM = A.confusion;
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const TEAL = "0891B2", CORAL = "DB2777", AMBER = "D97706", SLATE = "64748B", GREEN = "059669", RED = "DC2626", VIOLET = "7C3AED", LIGHT = "FFFFFF", WHITE = "FFFFFF", INK = "0F172A", BG = "F5F8FD";
const HF = "Calibri", BF = "Calibri", MONO = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: HF, bodyFontFace: BF };
pres.author = "Gladys Izuagbe"; pres.title = "CareConnect - Business Insights Presentation (BAN 6800 Module 5)";

let n = 0;
function glow(s, x, y, d, color, tr) { s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color, transparency: tr }, line: { color, transparency: 100 } }); }
function net(s, x, y, w, h, layers = [3, 5, 5, 3]) {
  const pts = layers.map((c, li) => Array.from({ length: c }, (_, k) => ({ x: x + (w * li) / (layers.length - 1), y: y + (h * (k + 0.5)) / c })));
  const cols = [TEAL, VIOLET, CORAL, GREEN];
  for (let l = 0; l < pts.length - 1; l++) for (const a of pts[l]) for (const b of pts[l + 1]) {
    const fx = a.y > b.y; s.addShape(pres.shapes.LINE, { x: a.x, y: Math.min(a.y, b.y), w: b.x - a.x, h: Math.abs(b.y - a.y) || 0.001, flipV: fx, line: { color: cols[l % 4], width: 0.6, transparency: 55 } });
  }
  pts.forEach((L, li) => L.forEach(p => { glow(s, p.x - 0.17, p.y - 0.17, 0.34, cols[li % 4], 75); s.addShape(pres.shapes.OVAL, { x: p.x - 0.07, y: p.y - 0.07, w: 0.14, h: 0.14, fill: { color: cols[li % 4] }, line: { color: WHITE, width: 0.5, transparency: 40 } }); }));
}
function slide(title, opts = {}) {
  const s = pres.addSlide(); n++;
  s.background = { color: BG };
  glow(s, 7.6, -1.4, 3.6, VIOLET, 88); glow(s, -1.2, 4.3, 3.2, TEAL, 88);
  if (opts.dark) { glow(s, 5.5, 0.6, 5, TEAL, 86); glow(s, 8, 3, 3.5, CORAL, 90); }
  if (title) {
    s.addText("// CARECONNECT  AI", { x: 0.5, y: 0.14, w: 4, h: 0.22, fontFace: MONO, fontSize: 9, color: TEAL, charSpacing: 3, margin: 0, isTextBox: true });
    s.addText(title, { x: 0.5, y: 0.34, w: 9, h: 0.72, fontFace: HF, fontSize: 22, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 1.1, h: 0.04, fill: { color: TEAL }, line: { color: TEAL, transparency: 100 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 1.6, y: 1.1, w: 0.5, h: 0.04, fill: { color: VIOLET }, line: { color: VIOLET, transparency: 100 } });
  }
  s.addText(String(n).padStart(2, "0"), { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontFace: MONO, fontSize: 10, color: TEAL, align: "right", margin: 0, isTextBox: true });
  if (opts.src) s.addText(opts.src, { x: 0.5, y: 5.25, w: 8.4, h: 0.25, fontSize: 10, color: SLATE, italic: true, margin: 0, isTextBox: true });
  return s;
}
function card(s, x, y, w, h, head, body, o = {}) {
  const ac = o.hc || TEAL;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: o.fill || LIGHT }, line: { color: ac, width: 0.75, transparency: 45 }, rectRadius: 0.1 });
  s.addShape(pres.shapes.RECTANGLE, { x: x + 0.02, y: y + 0.14, w: 0.05, h: Math.min(0.32, h - 0.28), fill: { color: ac }, line: { color: ac, transparency: 100 } });
  if (head) s.addText(head, { x: x + 0.18, y: y + 0.1, w: w - 0.3, h: 0.35, fontFace: HF, fontSize: o.hs || 14, bold: true, color: ac, margin: 0, isTextBox: true, valign: "top" });
  if (body) s.addText(body, { x: x + 0.18, y: y + (head ? 0.48 : 0.12), w: w - 0.32, h: h - (head ? 0.58 : 0.24), fontSize: o.bs || 12, color: o.bc || INK, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 3 });
}
function stat(s, x, y, w, big, label, color) {
  const c = color || TEAL;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x - 0.04, y: y - 0.04, w: w + 0.08, h: 1.33, fill: { color: c, transparency: 90 }, line: { color: c, transparency: 100 }, rectRadius: 0.12 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.25, fill: { color: LIGHT }, line: { color: c, width: 1, transparency: 35 }, rectRadius: 0.1 });
  s.addText(big, { x, y: y + 0.1, w, h: 0.65, fontFace: HF, fontSize: 36, bold: true, color: c, align: "center", margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.1, y: y + 0.75, w: w - 0.2, h: 0.45, fontSize: 11, color: INK, align: "center", margin: 0, isTextBox: true, valign: "top" });
}
const H = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: "0E7490", fill: { color: "E6F0FB" }, fontSize: 11, fontFace: BF, valign: "middle" }, o) });
const C = (t, o = {}) => ({ text: String(t), options: Object.assign({ fontSize: 11, color: INK, fill: { color: "FFFFFF" }, fontFace: BF, valign: "middle" }, o) });
const OK = (b, t) => C(t || (b ? "Yes" : "No"), { bold: true, color: b ? GREEN : RED });
function table(s, rows, x, y, w, colW, o = {}) {
  s.addTable(rows, { x, y, w, colW, border: { type: "solid", color: "D5DEEA", pt: 0.75 }, rowH: o.rowH || 0.34, autoPage: false, margin: [0.03, 0.08, 0.03, 0.08] });
}
const bullets = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < arr.length - 1 }, o) }));
const chartBase = { catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelColor: INK, valAxisLabelColor: SLATE, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, dataLabelFontSize: 11, dataLabelColor: INK, legendFontSize: 11, legendColor: INK, titleFontSize: 14, titleColor: TEAL, showTitle: true };
const ageOrder = ["child", "teen", "young_adult", "adult", "senior", "elderly"];
const ageLab = { child: "Child 0-12", teen: "Teen 13-18", young_adult: "Young adult 19-35", adult: "Adult 36-55", senior: "Senior 56-75", elderly: "Elderly 76+" };

// ═════════ MAIN DECK (15) ═════════
// 1 Title
let s = slide(null, { dark: true });
net(s, 6.0, 0.8, 3.5, 3.3, [3, 5, 6, 5, 2]);
s.addText("// BAN 6800  |  MODULE 5  |  EXPLAINABLE AI", { x: 0.6, y: 0.55, w: 5.6, h: 0.3, fontFace: MONO, fontSize: 9.5, color: TEAL, charSpacing: 2, margin: 0, isTextBox: true });
s.addText("CareConnect", { x: 0.6, y: 0.95, w: 5.6, h: 0.9, fontFace: HF, fontSize: 54, bold: true, color: INK, margin: 0, isTextBox: true });
s.addText("Predictive Engagement System", { x: 0.6, y: 1.8, w: 5.6, h: 0.55, fontFace: HF, fontSize: 26, bold: true, color: TEAL, margin: 0, isTextBox: true });
s.addText("What the no-show model found, why it decides the way it does, and how fairly it treats patients", { x: 0.6, y: 2.45, w: 5.2, h: 0.8, fontSize: 15, color: INK, margin: 0, isTextBox: true });
s.addText([{ text: "Gladys Izuagbe (ID 160948)", options: { bold: true, color: INK, breakLine: true } }, { text: "Nexford University  |  October 2026", options: { color: SLATE, breakLine: true } }, { text: "Client: CrescentCare Community Hospital Network", options: { color: SLATE } }], { x: 0.6, y: 3.35, w: 5.4, h: 0.8, fontSize: 13, margin: 0, isTextBox: true });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.3, w: 8.8, h: 0.8, fill: { color: "FFFFFF" }, line: { color: TEAL, width: 1, transparency: 40 }, rectRadius: 0.1 });
s.addText([{ text: "OBJECTIVE  ", options: { fontFace: MONO, bold: true, color: AMBER } }, { text: "Explain results, drivers, fairness and limits to non-technical stakeholders.   ", options: { color: INK, breakLine: true } }, { text: "LIVE DASHBOARD  ", options: { fontFace: MONO, bold: true, color: AMBER } }, { text: URL, options: { color: TEAL, hyperlink: { url: URL } } }], { x: 0.8, y: 4.32, w: 8.4, h: 0.76, fontSize: 12, margin: 0, isTextBox: true, valign: "middle" });
s.addNotes("Opening. Purpose: communicate findings neutrally. No recommendation is made in this deck, per the assignment scope. All figures come from the Module 3 pipeline and Module 4 model run on the real Kaggle data.");

// 2 Audience
s = slide("Who this is for, and what they need to decide");
card(s, 0.5, 1.2, 4.35, 1.75, "Primary audience", "Chief Operations Officer (capacity and budget)\nCare Coordination Manager (daily workflow)\nSocial Work Lead (support design and equity)", { bs: 13 });
card(s, 0.5, 3.1, 4.35, 1.95, "Ethics and compliance (Module 1)", "AI Ethics Committee (binding veto)\nPrivacy Officer (HIPAA / GDPR)\nPatient Advocate\nData Science Team (bias audits)", { bs: 13, fill: "FFF7E6", hc: AMBER });
card(s, 5.15, 1.2, 4.35, 3.85, "Questions these stakeholders need answered", "Can the scores be trusted enough to act on?\n\nHow much coordinator time would the flagged lists use?\n\nAre different patient groups treated equitably?\n\nWhat are the safeguards if the model drifts or misbehaves?\n\nWhat does the model not know?", { bs: 14 });
s.addNotes("Audience slide. These are decision needs, not recommendations. Ethics stakeholders come from Module 1 section 3.");

// 3 Business problem
s = slide("The problem: more than one in four booked slots goes unused", { src: "Sources: Module 1 Vision Document; Kaggle 'Medical Appointment No Shows' (Vianna, 2016); Module 4 hold-out set" });
stat(s, 0.5, 1.2, 2.1, "28%", "CrescentCare no-show rate (Module 1)", RED);
stat(s, 2.75, 1.2, 2.1, "72%", "Appointment slot fill rate (Module 1)", AMBER);
stat(s, 5.0, 1.2, 2.1, pct(D.overall, 1), "No-show rate in the public data the model learned from", TEAL);
stat(s, 7.25, 1.2, 2.25, "~$150B", "Estimated yearly U.S. cost of no-shows (Module 1)", SLATE);
table(s, [[H("Module 1 goal"), H("Target"), H("What this model's evidence can and cannot say")],
  [C("G1 Fewer no-shows"), C("28% to <15%"), C("Not testable yet: the model scores risk, it does not itself cut no-shows. Needs a pilot.")],
  [C("G2 Fuller clinics"), C("72% to >90%"), C("Not testable yet: depends on how staff act on flagged appointments.")],
  [C("G3 Equity (failure metric)"), C("Gap 18% to <5%"), C("Medicaid vs private is not in the data. Welfare-scholarship status is used as a stand-in.")],
  [C("G4 Patient trust"), C("Zero complaints"), C("Safeguards are designed (slide 13); no patients have been involved.")]],
  0.5, 2.75, 9.0, [2.1, 1.5, 5.4], { rowH: 0.44 });
s.addNotes("Business problem from Module 1. Note the training data's no-show rate (about 20%) is lower than CrescentCare's 28%; the model has not seen CrescentCare patients.");

// 4 Model plain language
s = slide("The model in plain language: a smoke detector for appointments");
s.addText("Every booked appointment gets a risk score from 0% to 100%: the model's estimate of the chance the patient will not turn up. It points a person to where a closer look may help. It is not a verdict on any patient.", { x: 0.5, y: 1.15, w: 9, h: 0.75, fontSize: 14, color: INK, margin: 0, isTextBox: true });
card(s, 0.5, 2.05, 4.35, 1.4, "What it does", `Scores appointments using ${D.n.toLocaleString("en-US")} past ones: how far ahead they were booked, past attendance, age, reminders and neighbourhood.`);
card(s, 5.15, 2.05, 4.35, 1.4, "Who acts on it", "Care coordinators and social workers. The model suggests; a person decides what, if anything, to offer.");
card(s, 0.5, 3.6, 4.35, 1.45, "What it cannot do", "Know why a patient may miss a visit, predict any one person with certainty, or replace a coordinator's judgement.", { fill: "FFF7E6", hc: AMBER });
card(s, 5.15, 3.6, 4.35, 1.45, "What it is built from", "A gradient-boosted decision-tree model (XGBoost), compared with two simpler models. All three reach similar accuracy.");
s.addNotes("Plain-language explanation. Avoid jargon: describe XGBoost only as a decision-tree model if asked.");

// 5 Key results
s = slide("Key results: useful ranking, but many flags are false alarms", { src: "Hold-out set of " + R.n_test.toLocaleString("en-US") + " appointments the model never saw during training" });
stat(s, 0.5, 1.2, 2.1, OP.roc_auc.toFixed(2), "Ranking ability (ROC-AUC): 0.50 is a coin flip", TEAL);
stat(s, 2.75, 1.2, 2.1, pct(OP.recall), "Of real no-shows, share the model flags", GREEN);
stat(s, 5.0, 1.2, 2.1, pct(OP.precision), "Of flagged appointments, share that are real no-shows", CORAL);
stat(s, 7.25, 1.2, 2.25, pct(OP.flag_rate), "Of all appointments flagged for outreach", SLATE);
table(s, [[H("Module 4 acceptance target"), H("Target"), H("Result"), H("Met?")],
  [C("ROC-AUC (ranking ability)"), C("at least 0.70"), C(OP.roc_auc.toFixed(3)), OK(OP.roc_auc >= 0.70)],
  [C("Recall (no-shows caught)"), C("at least 0.65"), C(OP.recall.toFixed(3)), OK(OP.recall >= 0.65)],
  [C("F1 (balance of caught vs correct)"), C("at least 0.55"), C(OP.f1.toFixed(3)), OK(OP.f1 >= 0.55)],
  [C("Fairness: gender gap"), C("at most 0.05"), C(F.Gender.eod.toFixed(3)), OK(F.Gender.pass_eod)],
  [C("Fairness: age-group gap"), C("at most 0.05"), C(F.age_band.eod.toFixed(3)), OK(F.age_band.pass_eod)]],
  0.5, 2.65, 9.0, [3.6, 1.9, 1.7, 1.8], { rowH: 0.33 });
s.addText(`The flagging cut-off (${OP.threshold.toFixed(2)}) was chosen on training data only, to catch about two in three no-shows. Using the default 50% cut-off instead, the model would catch only ${pct(OP5.recall)}.`, { x: 0.5, y: 4.8, w: 9, h: 0.4, fontSize: 11, color: SLATE, margin: 0, isTextBox: true });
s.addNotes(`Be direct: AUC and recall targets are met; F1 is ${OP.f1.toFixed(2)} versus 0.55 target; age fairness fails on the standard rule. Default 0.5 threshold recall ${pct(OP5.recall,1)}.`);

// 6 Visual summary
s = slide("Visual summary: higher scores and longer waits mean more missed visits", { src: "Left: all " + D.n.toLocaleString("en-US") + " appointments. Right: unseen test appointments in 10 equal groups by score." });
s.addChart(pres.charts.BAR, [{ name: "% missed", labels: D.lead_time.map(d => d.bucket), values: D.lead_time.map(d => +(d.mean * 100).toFixed(1)) }],
  Object.assign({}, chartBase, { x: 0.4, y: 1.15, w: 4.7, h: 3.95, barDir: "col", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"", showLegend: false, title: "Missed visits by days booked ahead (%)", valAxisMaxVal: 40, valAxisLabelFormatCode: "0\"%\"" }));
s.addChart(pres.charts.BAR, [{ name: "Actually missed", labels: A.deciles.map(d => String(d.d + 1)), values: A.deciles.map(d => +(d.actual * 100).toFixed(1)) }, { name: "Model's average score", labels: A.deciles.map(d => String(d.d + 1)), values: A.deciles.map(d => +(d.pred * 100).toFixed(1)) }],
  Object.assign({}, chartBase, { x: 5.2, y: 1.15, w: 4.4, h: 3.95, barDir: "col", chartColors: [TEAL, CORAL], showValue: false, showLegend: true, legendPos: "b", title: "By score group, 1 lowest to 10 highest (%)", valAxisLabelFormatCode: "0\"%\"" }));
s.addNotes(`Left: same-day bookings are missed ${pct(D.lead_time[0].mean,1)} of the time; 31+ days ahead ${pct(D.lead_time[5].mean)}. Right: lowest score group missed ${pct(A.deciles[0].actual)}, highest ${pct(A.deciles[9].actual)}; scores track real outcomes closely.`);

// 7 Drivers
s = slide("What drives the predictions: mostly how far ahead the visit was booked", { src: "SHAP analysis of 4,000 sampled test appointments (exact tree explanation)" });
const PL = { lead_time_days: "Days booked ahead", Age: "Patient age", prior_noShow_rate: "Share of past visits missed", SMS_received: "Reminder SMS sent", prior_appointments: "Number of past visits", appointment_month: "Month of visit", Neighbourhood: "Neighbourhood", prior_noShows: "Number of past misses", age_band: "Age group", Scholarship: "Welfare scholarship", Gender: "Gender" };
const top = Object.entries(A.shap.share).slice(0, 8);
s.addChart(pres.charts.BAR, [{ name: "Share of influence", labels: top.map(t => PL[t[0]] || t[0]).reverse(), values: top.map(t => +(t[1] * 100).toFixed(1)).reverse() }],
  Object.assign({}, chartBase, { x: 0.4, y: 1.15, w: 5.6, h: 3.95, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"", showLegend: false, title: "Share of the model's decision-making (%)", valAxisMinVal: 0, valAxisMaxVal: 75, valAxisLabelFormatCode: "0\"%\"" }));
card(s, 6.2, 1.2, 3.3, 3.85, "How to read this", `Booking lead time carries about ${pct(A.shap.share.lead_time_days)} of the influence. Age and past missed visits add roughly ${pct(A.shap.share.Age + A.shap.share.prior_noShow_rate)} more.\n\nNeighbourhood carries only about ${pct(A.shap.share.Neighbourhood)}, so it is a minor factor here.\n\nThese are patterns in past data, not proven causes.`, { bs: 12 });
s.addNotes("SHAP shares are each factor's average absolute push on the score, as a share of the total. Neighbourhood was described as a transit proxy in earlier modules; in this run it is small.");

// 8 Examples
s = slide("Three example patients, explained in plain language", { src: "Real, unseen test-set appointments; reasons come from the model's own SHAP breakdown" });
const PLN = { lead_time_days: "How far ahead it was booked", prior_noShow_rate: "Share of past visits missed", prior_noShows: "Past missed visits", SMS_received: "Reminder SMS status", appointment_month: "Month of visit", Age: "Patient age", age_band: "Age group", prior_appointments: "Number of past visits", Neighbourhood: "Neighbourhood" };
const colr = { high: RED, medium: AMBER, low: GREEN };
A.samples.forEach((p, i) => {
  const x = 0.5 + i * 3.05, f = p.features;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.15, w: 2.9, h: 3.95, fill: { color: LIGHT }, line: { color: colr[p.tier], pt: 2 }, rectRadius: 0.08 });
  s.addText(`${p.tier[0].toUpperCase() + p.tier.slice(1)} risk: ${pct(p.prob)}`, { x: x + 0.15, y: 1.22, w: 2.6, h: 0.4, fontFace: HF, fontSize: 18, bold: true, color: colr[p.tier], margin: 0, isTextBox: true });
  s.addText(`${f.Gender === "F" ? "Woman" : "Man"}, age ${f.Age}. Booked ${f.lead_time_days} days ahead. ${f.prior_noShows} of ${f.prior_appointments} past visits missed. SMS ${f.SMS_received ? "sent" : "not sent"}.`, { x: x + 0.15, y: 1.65, w: 2.6, h: 0.85, fontSize: 11, color: INK, margin: 0, isTextBox: true, valign: "top" });
  s.addText("Main reasons", { x: x + 0.15, y: 2.55, w: 2.6, h: 0.25, fontSize: 12, bold: true, color: TEAL, margin: 0, isTextBox: true });
  s.addText(p.top.slice(0, 3).map((t, j) => ({ text: `${t[1] > 0 ? "Raises" : "Lowers"} risk: ${PLN[t[0]] || t[0]}`, options: { bullet: true, breakLine: j < 2 } })), { x: x + 0.15, y: 2.82, w: 2.6, h: 1.15, fontSize: 11, color: INK, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 3 });
  s.addText(`What happened: ${p.actual ? "did not attend" : "attended"}`, { x: x + 0.15, y: 4.55, w: 2.6, h: 0.4, fontSize: 11, italic: true, color: SLATE, margin: 0, isTextBox: true, valign: "bottom" });
});
s.addNotes("Examples were chosen to span the three risk tiers and different age groups, not to flatter the model. The medium example attended; the model is only a probability.");

// 9 What-if
s = slide("What-if analysis: changing one input moves the score only modestly", { src: "Average change in predicted risk across all unseen test appointments, in percentage points. The dashboard lets you try any patient." });
const wi = Object.entries(A.whatif_avg_pp);
const wiLab = { 0: "Booked 7 days ahead at most", 1: "Booked 14 days ahead at most", 2: "Everyone marked 'SMS sent'", 3: "Nobody marked 'SMS sent'", 4: "No past missed visits" };
s.addChart(pres.charts.BAR, [{ name: "Points", labels: wi.map((_, i) => wiLab[i]).reverse(), values: wi.map(w => +w[1].toFixed(1)).reverse() }],
  Object.assign({}, chartBase, { x: 0.4, y: 1.15, w: 5.6, h: 3.95, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0", showLegend: false, title: "Average change in risk score (points)", catAxisLabelPos: "low", valAxisMinVal: -2.6, valAxisMaxVal: 2, valAxisLabelFormatCode: "0.0" }));
card(s, 6.2, 1.2, 3.3, 3.85, "What this shows", `Capping lead time at 7 days lowers the average score by ${Math.abs(wi[0][1]).toFixed(1)} points; removing past misses by ${Math.abs(wi[4][1]).toFixed(1)}.\n\nThe SMS result runs against intuition in the raw data (SMS patients miss more: ${pct(D.sms[1].mean)} vs ${pct(D.sms[0].mean)}) because reminders go mostly to visits booked further ahead.\n\nThese show how the score moves, not that real behaviour will change.`, { bs: 11.5 });
s.addNotes("What-if effects are on the model's score, holding other inputs fixed. They are associations. A randomised pilot would be needed to show real-world effect.");

// 10 Fairness
s = slide("Fairness summary: gender is even, but older patients are flagged far less often", { src: "Standard flagging rule on unseen test data. Charter limit: gaps of 0.05 or less." });
const fr = (k, nm) => [C(nm), C(F[k].dpd.toFixed(3)), C(F[k].eod.toFixed(3)), C(F[k].di_ratio.toFixed(2)), OK(F[k].pass_dpd && F[k].pass_eod && F[k].pass_di)];
table(s, [[H("Groups compared"), H("Flagging gap"), H("Error-rate gap"), H("Lowest ÷ highest flag rate"), H("Within limit?")],
  fr("Gender", "Gender"), fr("age_band", "Age group"), fr("Scholarship_grp", "Welfare scholarship (socio-economic stand-in)")],
  0.5, 1.1, 9.0, [3.4, 1.2, 1.3, 1.9, 1.2], { rowH: 0.34 });
const ab = ageOrder.map(k => F.age_band.groups[k]);
s.addChart(pres.charts.BAR, [{ name: "Actually missed", labels: ageOrder.map(k => ageLab[k]), values: ab.map(g => +(g.no_show_rate * 100).toFixed(1)) }, { name: "Flagged by the model", labels: ageOrder.map(k => ageLab[k]), values: ab.map(g => +(g.flag_rate * 100).toFixed(1)) }],
  Object.assign({}, chartBase, { x: 0.4, y: 2.65, w: 5.8, h: 2.6, barDir: "col", chartColors: [SLATE, TEAL], showValue: false, showLegend: true, legendPos: "b", title: "By age group (%)", catAxisLabelFontSize: 9, valAxisLabelFormatCode: "0\"%\"" }));
card(s, 6.35, 2.95, 3.15, 2.2, "In plain terms", `Seniors miss ${pct(F.age_band.groups.senior.no_show_rate)} of visits but are flagged ${pct(F.age_band.groups.senior.flag_rate)} of the time; young adults miss ${pct(F.age_band.groups.young_adult.no_show_rate)} and are flagged ${pct(F.age_band.groups.young_adult.flag_rate)}. The model catches ${pct(F.age_band.groups.senior.tpr)} of senior no-shows vs ${pct(F.age_band.groups.young_adult.tpr)} of young adults'.`, { bs: 11 });
s.addNotes("Gender passes. Age group fails by a wide margin, and the welfare-scholarship stand-in also fails. Under the Ethical Charter, a breach of the 5% limit is a halt condition. This is reported plainly.");

// 11 Mitigation
s = slide("Bias mitigation: what was found, what was tried, what it cost", { src: "Fairlearn ThresholdOptimizer fitted on training-set predictions; all results on unseen test data" });
const mrow = (k, nm) => { const m = MIT[k]; const ok = x => OK(x <= 0.05, x.toFixed(3)); return [C(nm), C(pct(m.recall)), C(pct(m.precision)), C(pct(m.flag_rate)), ok(m.age_band.eod), ok(m.Gender.eod), ok(m.Scholarship.eod)]; };
table(s, [[H("Approach"), H("No-shows caught"), H("Flags that are real"), H("Appointments flagged"), H("Age gap"), H("Gender gap"), H("Scholarship gap")],
  mrow("before", "Standard single cut-off"), mrow("A_group_thresholds", "A. Separate cut-off per age group"), mrow("B_equalized_odds", "B. Fairlearn equalized-odds rule")],
  0.5, 1.2, 9.0, [2.7, 1.1, 1.1, 1.2, 0.95, 0.95, 1.0], { rowH: 0.5 });
card(s, 0.5, 3.3, 2.9, 1.8, "Found", "The model flags young patients much more than older ones, and catches far fewer of the older patients' no-shows.", { bs: 12 });
card(s, 3.55, 3.3, 2.9, 1.8, "Tried", `A narrows the age gap to ${MIT.A_group_thresholds.age_band.eod.toFixed(2)} but not under 0.05. B brings every gap under 0.05.`, { bs: 12 });
card(s, 6.6, 3.3, 2.9, 1.8, "Cost", `B flags ${pct(MIT.B_equalized_odds.flag_rate)} of appointments instead of ${pct(MIT.before.flag_rate)}, and part of its rule is randomised.`, { bs: 12, fill: "FFF7E6", hc: AMBER });
s.addNotes("This is a trade-off between equity and coordinator workload. The deck lays it out for the Ethics Committee and leadership; it does not choose.");

// 12 Limitations
s = slide("What the model cannot do: limits stakeholders should know first");
const lim = [["Not trained on CrescentCare", `Data are Brazilian public-health visits from 2016, about six weeks. The hospital's 28% rate is above the data's ${pct(D.overall)}.`],
  ["Often wrong about individuals", `Of ${(CM.tp + CM.fp).toLocaleString("en-US")} flagged on the test set, ${CM.fp.toLocaleString("en-US")} attended anyway; ${CM.fn.toLocaleString("en-US")} real no-shows were missed.`],
  ["Does not know why", "Transport, childcare, cost and health literacy are not in the data; reasons shown are statistical patterns."],
  ["Misses some Module 4 targets", `F1 is ${OP.f1.toFixed(2)} (target 0.55). The standard rule fails the age-group fairness limit.`],
  ["Equity goal not measurable", "Module 1's Medicaid-vs-private gap needs an insurance field the dataset lacks."],
  ["Associations, not causes", "A short window cannot show seasonal change or prove that any action changes attendance."]];
lim.forEach((l, i) => card(s, 0.5 + (i % 3) * 3.05, 1.2 + Math.floor(i / 3) * 1.95, 2.9, 1.8, l[0], l[1], { bs: 11, hs: 13 }));
s.addNotes("Limitations are stated before any suggestion to act. The training data difference from CrescentCare is the most important caveat.");

// 13 Transparency
s = slide("Transparency and ethics: what the system will and will not do", { src: "Ethical Charter and governance from Module 1; privacy plan from Modules 2-3" });
card(s, 0.5, 1.2, 4.35, 1.6, "Transparency statement", "CareConnect is decision support only. It never denies, delays or restricts care, never penalises a patient, and never schedules or contacts anyone automatically.", { fill: "E8F8F2", hc: GREEN, bs: 12 });
card(s, 5.15, 1.2, 4.35, 1.6, "Can do / cannot do", "Can: rank appointments by risk and show reasons.\nCannot: diagnose, explain causes, or decide who is offered what.", { bs: 12 });
table(s, [[H("Charter principle"), H("How it appears in this work")],
  [C("Beneficence / Non-maleficence"), C("Outputs trigger supportive offers only; fairness gaps are published, including the failures.")],
  [C("Autonomy / Privacy"), C("IDs are hashed, appointment IDs dropped, opt-out honoured, no personal data in the repository.")],
  [C("Justice"), C("The equity limit (0.05) is a halt condition; the age-group breach is reported, not hidden.")],
  [C("Human accountability"), C("Every outreach is a documented human decision; the Ethics Committee holds a veto.")]],
  0.5, 2.95, 9.0, [2.6, 6.4], { rowH: 0.42 });
s.addNotes("Governance from Module 1: AI Ethics Committee with binding veto, Privacy Officer, Patient Advocate. Charts in this deck start at zero and show every group, so they do not exaggerate or hide differences.");

// 14 Implications
s = slide("Business implications: what the findings mean for operations", { src: "Per 1,000 booked appointments, using the unseen test-set rates. Illustrative; CrescentCare's own rates may differ." });
const per = 1000, flagged = Math.round(per * OP.flag_rate), caught = Math.round(flagged * OP.precision), realNS = Math.round(per * A.base_rate), fB = Math.round(per * MIT.B_equalized_odds.flag_rate), cB = Math.round(fB * MIT.B_equalized_odds.precision);
table(s, [[H("Finding"), H("What it means for the business")],
  [C("Lead time is the biggest factor"), C(`Visits booked a month or more ahead are missed ${pct(D.lead_time[5].mean)} of the time, same-day ones ${pct(D.lead_time[0].mean)}. Booking windows are linked to attendance in the data.`)],
  [C("Flagged lists include many false alarms"), C(`Per 1,000 appointments, about ${realNS} are real no-shows. The standard rule flags ${flagged}, of which about ${caught} are real; roughly ${flagged - caught} flagged patients would have attended anyway.`)],
  [C("Fairness fixes change workload"), C(`The equalized-odds rule flags about ${fB} per 1,000 (about ${cB} real), so outreach volume rises by roughly ${fB - flagged} appointments per 1,000.`)],
  [C("Equity and efficiency pull apart"), C("The rule that catches no-shows most efficiently is the one that flags older patients least.")],
  [C("Impact is unproven"), C("Savings depend on whether outreach changes attendance, which only a pilot can show.")]],
  0.5, 1.15, 9.0, [2.9, 6.1], { rowH: 0.6 });
s.addNotes("These are implications, not recommendations, as the assignment requires. No dollar ROI is stated because no outreach effect has been measured.");

// 15 Next steps
s = slide("Possible next steps for stakeholder discussion (not a recommendation)");
const ns = [["Local validation", "Test the model on a CrescentCare EHR sample (Module 2 assumption A-1) before relying on any score."],
  ["Fairness decision", "The Ethics Committee weighs options A and B (slide 11) against coordinator capacity."],
  ["Measure the equity goal", "Add an insurance-type field so Module 1's Medicaid-vs-private gap can be tracked directly."],
  ["Design a pilot", "Define what outreach is offered, to whom, and how attendance and fairness are measured."],
  ["Close the F1 gap", "Review whether the 0.55 F1 target is realistic for this data, or improve features."],
  ["Monitor after launch", "Track score drift and group gaps monthly, with the halt rule from the Charter."]];
ns.forEach((x, i) => card(s, 0.5 + (i % 3) * 3.05, 1.15 + Math.floor(i / 3) * 1.55, 2.9, 1.42, x[0], x[1], { bs: 11, hs: 13 }));
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.3, w: 9, h: 0.8, fill: { color: "FFFFFF" }, line: { color: TEAL, width: 1, transparency: 40 }, rectRadius: 0.1 });
s.addText([{ text: "Explore it yourself:  ", options: { bold: true, color: AMBER } }, { text: URL, options: { color: TEAL, hyperlink: { url: URL } } }, { text: "   |   Six pages: results, drivers, risk explorer with what-if, ethical compliance dashboard, limitations.", options: { color: SLATE } }], { x: 0.7, y: 4.32, w: 8.6, h: 0.76, fontSize: 12, margin: 0, isTextBox: true, valign: "middle" });
s.addNotes("These are options for discussion. This deck intentionally contains no formal recommendation or executive summary.");

// ═════════ APPENDIX ═════════
s = slide(null, { dark: true });
net(s, 6.3, 1.2, 3.2, 3, [2, 4, 4, 2]);
s.addText("// APPENDIX", { x: 0.6, y: 1.5, w: 5, h: 0.3, fontFace: MONO, fontSize: 12, color: TEAL, charSpacing: 3, margin: 0, isTextBox: true });
s.addText("Q&A backup", { x: 0.6, y: 1.9, w: 5.6, h: 0.9, fontFace: HF, fontSize: 48, bold: true, color: INK, margin: 0, isTextBox: true });
s.addText("Technical detail for questions: data, method, models, thresholds, fairness detail, API, reproducibility, references and AI disclosure.", { x: 0.6, y: 2.9, w: 5.4, h: 1.2, fontSize: 15, color: SLATE, margin: 0, isTextBox: true });

// A1 data & pipeline
s = slide("A1. Data and pipeline: what the Module 3 code did on the real file", { src: "Module 3 pipeline stages run on the public Kaggle file; Prefect/Great Expectations wrapper not executed (orchestration only)" });
table(s, [[H("Step"), H("Result")],
  [C("Extract"), C("110,527 rows, 14 columns, schema matched")],
  [C("Clean"), C("13 rows removed: 8 invalid ages, 5 scheduled after the visit, 0 duplicate appointment IDs. 110,514 rows remain.")],
  [C("Anonymize"), C("Appointment ID dropped; patient ID salted-hash pseudonym; 75 neighbourhoods kept")],
  [C("Features"), C("Lead time, day/month, age band, prior visits, prior misses, prior miss rate (chronological, current visit excluded)")],
  [C("Bias gate (5 checks)"), C("All pass: gender 65/35, six age bands each over 5,000, top neighbourhood 7%, no-show ratio by gender 1.02")]],
  0.5, 1.2, 9.0, [1.9, 7.1], { rowH: 0.55 });
s.addNotes("Module 4's report states 110,527 records after cleaning and 21 features; the real pipeline output is 110,514 rows.");

// A2 method
s = slide("A2. Method: split, features, tuning and one important fix", { src: "Code: run_real.py in the Module 4 package" });
card(s, 0.5, 1.15, 4.35, 3.95, "Setup", `Stratified 80/20 split (seed 42): ${R.n_train.toLocaleString("en-US")} train, ${R.n_test.toLocaleString("en-US")} test.\nFeatures: gender, age, neighbourhood, welfare scholarship, hypertension, diabetes, alcoholism, disability level, SMS, lead time, day, month, age group, three prior-visit features.\nSame grids as Module 4 for all three models; tuning used a 25,000-row training sample with 3-fold cross-validation for speed, then the best settings were refitted on the full training set. The test set was used once.`, { bs: 11.5 });
card(s, 5.15, 1.15, 4.35, 1.9, "Fix: target leakage", "The Module 3 table keeps the raw 'No-show' text column. Module 4's train.py only dropped the numeric flag, so 'No-show' would have been fed to the model as an input. It was removed, along with raw timestamps and the patient hash.", { fill: "FFF7E6", hc: AMBER, bs: 11.5 });
card(s, 5.15, 3.2, 4.35, 1.9, "Fix: cut-off", `At the default 50% cut-off XGBoost catches only ${pct(OP5.recall,1)} of no-shows. A cut-off of ${OP.threshold.toFixed(3)} was chosen from 3-fold out-of-fold training predictions (recall at least 0.65), never from test data.`, { bs: 11.5 });

// A3 model comparison
s = slide("A3. Model comparison on the unseen test set", { src: "Baselines: most-frequent and stratified-random dummy classifiers" });
const mm = (nm, o, t) => [C(nm), C(t), C(o.roc_auc.toFixed(3)), C(o.recall.toFixed(3)), C(o.precision.toFixed(3)), C(o.f1.toFixed(3)), C(pct(o.flag_rate))];
const rows = [[H("Model"), H("Cut-off"), H("ROC-AUC"), H("Recall"), H("Precision"), H("F1"), H("Flagged")]];
["logistic_regression", "random_forest", "xgboost"].forEach(k => { const nm = { logistic_regression: "Logistic regression", random_forest: "Random forest", xgboost: "XGBoost (used)" }[k]; rows.push(mm(nm, A.operating[k].at_recall_target, A.operating[k].at_recall_target.threshold.toFixed(2))); });
rows.push([C("XGBoost at default cut-off"), C("0.50"), C(OP5.roc_auc.toFixed(3)), C(OP5.recall.toFixed(3)), C(OP5.precision.toFixed(3)), C(OP5.f1.toFixed(3)), C(pct(OP5.flag_rate))]);
rows.push([C("Majority-class baseline"), C("n/a"), C("0.500"), C("0.000"), C("0.000"), C("0.000"), C("0%")]);
rows.push([C("Random (stratified) baseline"), C("n/a"), C(R.baselines.stratified.roc_auc.toFixed(3)), C(R.baselines.stratified.recall.toFixed(3)), C(R.baselines.stratified.precision.toFixed(3)), C(R.baselines.stratified.f1.toFixed(3)), C("n/a")]);
table(s, rows, 0.5, 1.2, 9.0, [3.0, 0.9, 1.1, 1.0, 1.1, 0.9, 1.0], { rowH: 0.4 });
s.addText(`Random forest scored marginally higher on ROC-AUC (${R.models.random_forest.metrics.roc_auc.toFixed(3)} vs ${R.models.xgboost.metrics.roc_auc.toFixed(3)}); the three models are close. XGBoost is used throughout because it is the Module 4 API model.`, { x: 0.5, y: 4.55, w: 9, h: 0.5, fontSize: 12, color: INK, margin: 0, isTextBox: true });

// A4 confusion
s = slide("A4. What the standard flagging rule does to 22,103 unseen appointments", { src: `Cut-off ${OP.threshold.toFixed(3)}` });
table(s, [[H(""), H("Flagged"), H("Not flagged")], [C("Missed the visit", { bold: true }), C(CM.tp.toLocaleString("en-US") + " caught", { color: GREEN, bold: true }), C(CM.fn.toLocaleString("en-US") + " missed by model", { color: RED, bold: true })],
  [C("Attended", { bold: true }), C(CM.fp.toLocaleString("en-US") + " false alarms", { color: CORAL, bold: true }), C(CM.tn.toLocaleString("en-US") + " correctly not flagged", { color: GREEN, bold: true })]],
  0.5, 1.3, 5.2, [1.6, 1.8, 1.8], { rowH: 0.65 });
card(s, 6.0, 1.3, 3.5, 3.6, "Risk tiers (test set)", "High (score 40%+): about 5% of appointments, 53% actually missed.\nMedium (cut-off to 40%): about 34% of appointments, 32% missed.\nLow (below cut-off): about 61% of appointments, 11% missed.\nOverall rate: 20%.", { bs: 12 });
s.addText("Across the ten score groups the model's average score tracks the actual miss rate closely (slide 6), so the scores behave like genuine probabilities.", { x: 0.5, y: 3.5, w: 5.2, h: 0.9, fontSize: 12, color: INK, margin: 0, isTextBox: true });

// A5 fairness detail
s = slide("A5. Fairness detail by age group (standard rule)", { src: "Flag rate = share flagged. Catch rate = share of real no-shows flagged. False alarm = share of attendees flagged." });
const fr5 = [[H("Age group"), H("Appointments"), H("Actual no-show"), H("Flagged"), H("Catch rate"), H("False alarm")]];
ageOrder.forEach(k => { const g = F.age_band.groups[k]; fr5.push([C(ageLab[k]), C(g.n.toLocaleString("en-US")), C(pct(g.no_show_rate, 1)), C(pct(g.flag_rate, 1)), C(pct(g.tpr, 1)), C(pct(g.fpr, 1))]); });
table(s, fr5, 0.5, 1.2, 9.0, [2.4, 1.4, 1.5, 1.2, 1.3, 1.2], { rowH: 0.38 });
s.addText(`Gender: women flagged ${pct(F.Gender.groups.F.flag_rate, 1)}, men ${pct(F.Gender.groups.M.flag_rate, 1)}; catch rates ${pct(F.Gender.groups.F.tpr, 1)} and ${pct(F.Gender.groups.M.tpr, 1)}. Scholarship: flagged ${pct(F.Scholarship_grp.groups["Bolsa Familia scholarship"].flag_rate, 1)} with vs ${pct(F.Scholarship_grp.groups["No scholarship"].flag_rate, 1)} without. Metrics computed with Fairlearn; limit 0.05 from Charter Principle 2.`, { x: 0.5, y: 4.0, w: 9, h: 0.9, fontSize: 12, color: INK, margin: 0, isTextBox: true });

// A6 mitigation detail
s = slide("A6. Mitigation detail: per-age-group cut-offs and error rates", { src: "Option A cut-offs set on training out-of-fold predictions to give each age group about 65% recall" });
const th = MIT.A_group_thresholds.thresholds;
const r6 = [[H("Age group"), H("Option A cut-off"), H("A: flagged"), H("B: flagged"), H("B: catch rate"), H("B: false alarm")]];
ageOrder.forEach(k => { const b = MIT.B_equalized_odds.age_groups[k], a = MIT.A_group_thresholds.age_groups[k]; r6.push([C(ageLab[k]), C(th[k].toFixed(3)), C(pct(a.flag_rate)), C(pct(b.flag_rate)), C(pct(b.tpr)), C(pct(b.fpr))]); });
table(s, r6, 0.5, 1.2, 9.0, [2.4, 1.5, 1.3, 1.3, 1.3, 1.2], { rowH: 0.38 });
s.addText("Option B uses randomised decisions within group cut-offs, which a hospital would need to justify to patients. Option A is deterministic and explainable but leaves an age gap above the limit. Gender and scholarship gaps are under 0.05 under B.", { x: 0.5, y: 4.0, w: 9, h: 0.9, fontSize: 12, color: INK, margin: 0, isTextBox: true });

// A7 API
s = slide("A7. API demo: FastAPI scoring service (api_demo.py)", { src: "Endpoints: GET /health, GET /model_info, POST /predict. Run: python api_demo.py" });
s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 5.2, h: 3.85, fill: { color: "050813" }, line: { color: TEAL, width: 1, transparency: 50 } });
s.addText(`POST /predict   (patient A)\n{\n  "no_show_probability": 0.5294,\n  "risk_tier": "high",\n  "flagged_for_outreach": true,\n  "reason_codes": [\n    "Days between booking and\n     appointment: raises risk",\n    "Share of past appointments\n     missed: raises risk",\n    "Number of past missed\n     appointments: raises risk"\n  ]\n}`, { x: 0.7, y: 1.3, w: 4.9, h: 3.7, fontFace: "Courier New", fontSize: 10.5, color: GREEN, margin: 0, isTextBox: true, valign: "top" });
card(s, 5.9, 1.2, 3.6, 3.85, "What the demo shows", "Starts the service, checks health and model info, then scores three patients: A (long lead time, past misses): high; B (mid lead time, one past miss): medium; C (same-day booking, reliable): low.\n\nA request with missing fields returns a clear 422 error.\n\nEvery response carries the note that a human decides.", { bs: 11.5 });

// A8 discrepancies
s = slide("A8. Differences between earlier modules and the real run", { src: "Noted so the final project integrates consistent figures" });
table(s, [[H("Earlier statement"), H("What the real run shows")],
  [C("Module 4: 110,527 records after cleaning"), C("110,514 after the documented cleaning rules (13 removed)")],
  [C("Module 4: results 'populated from MLflow'"), C(`Real values: AUC ${OP.roc_auc.toFixed(3)}, recall ${OP.recall.toFixed(3)}, precision ${OP.precision.toFixed(3)}, F1 ${OP.f1.toFixed(3)}`)],
  [C("Modules 3-4: bias checks 'expected', fairness assumed to pass"), C("Data checks pass on the real file; the model fails age-group and scholarship fairness limits")],
  [C("Module 2: ED wait-time and churn text"), C("Left over from another project; this work uses the no-show framing only")],
  [C("Module 1 G3: Medicaid vs private gap"), C("Not in the dataset; welfare scholarship used as a stand-in and labelled as such")],
  [C("API tiers fixed at 0.4 / 0.7"), C("Scores rarely exceed 0.5, so tiers are now cut-off to 0.40 (medium) and 0.40+ (high)")]],
  0.5, 1.15, 9.0, [3.7, 5.3], { rowH: 0.55 });

// A9 references
s = slide("A9. References (NXU / APA 7th)");
const refs = ["Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., ... Herrera, F. (2020). Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion, 58, 82-115. https://doi.org/10.1016/j.inffus.2019.12.012",
  "Bird, S., Dudík, M., Edgar, R., Horn, B., Lutz, R., Milan, V., ... Walker, K. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI (Technical Report MSR-TR-2020-32). Microsoft.",
  "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785",
  "Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems 29 (pp. 3315-3323).",
  "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems 30 (pp. 4765-4774).",
  "Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6), 1-35. https://doi.org/10.1145/3457607",
  "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... Gebru, T. (2019). Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596",
  "Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464), 447-453. https://doi.org/10.1126/science.aax2342",
  "Vianna, J. (2016). Medical appointment no shows [Data set]. Kaggle. https://www.kaggle.com/datasets/joniarroba/noshowappointments"];
s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })), { x: 0.5, y: 1.1, w: 9, h: 4.1, fontSize: 11, color: INK, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 5 });

// A10 AI disclosure
s = slide("A10. AI disclosure and code repository");
card(s, 0.5, 1.2, 9.0, 1.9, "AI Disclosure Statement", "This presentation, the dashboard code and the analysis scripts were developed with the assistance of Artificial Intelligence tools for structure, code drafting and visual design. The modelling run, the interpretation of results, the fairness findings and the decision to report failures plainly were reviewed and validated by the author. The official AI Disclosure Form accompanies this submission.", { bs: 12.5 });
card(s, 0.5, 3.25, 9.0, 1.8, "Code and live links", `Dashboard: ${URL}\nRepository: the Module 2 GitHub repository, with the Module 5 folder added (src/app.py, artifacts/, requirements.txt, api_demo.py, src/api/main.py)\nRebuild locally: pip install -r requirements.txt, then streamlit run src/app.py`, { bs: 12 });

pres.writeFile({ fileName: "/home/claude/pk/CareConnect_Module5_Presentation.pptx" }).then(() => console.log("written", n, "slides"));
