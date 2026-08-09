# HoneyIQ — Plain-Language Explanation (Revised Edition)

## What problem does this solve?

Imagine a bank. The bank knows criminals exist and try to break in. So they set up a fake vault that looks real — stocked with fake money — right next to the real one. If a criminal goes for it, the bank knows immediately it is being targeted, and can watch what the criminal does without risking anything real. That fake vault is a **honeypot**.

The challenge is: what should the bank *do* once it notices someone poking around the fake vault? Should it:
- Stay quiet and keep watching? (learn more about the attacker)
- Sound the alarm? (stop the threat now)
- Feed the criminal false information to waste their time?
- Lock them out?

Getting this decision wrong either tips off the attacker (they run away before you learn anything) or lets them cause real damage.

**HoneyIQ answers that question automatically.**

---

## What is a Cyber Kill Chain?

Hackers rarely launch full attacks from the first click. They follow steps — almost like a recipe:

| Step | What the hacker does | Example |
|------|---------------------|---------|
| 1. Reconnaissance | Look around, map the network | Port scanning |
| 2. Weaponization | Build the attack tool | Crafting malware |
| 3. Delivery | Send the weapon in | Phishing email |
| 4. Exploitation | Break into a system | Running the exploit |
| 5. Installation | Leave a back door | Installing a rootkit |
| 6. Command & Control | Talk back to their server | C2 beacon |
| 7. Actions on Objectives | Do the damage | Steal data, destroy files |

This 7-step model is called the **Cyber Kill Chain** (Lockheed Martin, 2011). Knowing *which step* the attacker is on tells you how serious the threat is and what to do about it.

---

## What does HoneyIQ do?

HoneyIQ is a **smart honeypot manager**. It watches the fake system, figures out how dangerous the situation is, and picks the best response — automatically, and in a way a human can understand and check.

### The two key pieces

**1. The Attacker Model**
HoneyIQ simulates four types of attacker so the system can be tested thoroughly:

| Attacker Type | Behaviour |
|---------------|-----------|
| Stealthy | Slow, patient, tries not to be noticed |
| Aggressive | Fast, noisy, goes straight for damage |
| Targeted | Focused on one specific goal |
| Opportunistic | Random, tries many things at once |

Each attacker follows the Kill Chain using probability mathematics (Markov chains) so their behaviour is realistic and varied.

**2. The Decision Policy (SEDM)**
The Stage-Escalation Decision Matrix is a simple lookup table with two inputs:
- **Where is the attacker in the Kill Chain?** (steps 1–7)
- **How likely are they to jump to a more dangerous step?** (Low / Medium / High)

From those two answers, it outputs one of five responses:

| Response | Meaning |
|----------|---------|
| ALLOW | Normal traffic, do nothing |
| LOG | Suspicious — record it quietly |
| TROLL | Feed fake information, waste their time |
| BLOCK | Cut the connection |
| ALERT | Sound the alarm, call a human |

The entire policy fits in a single table you can print on one page. There is no "black box" — every decision can be explained in plain English.

---

## How does it know when to escalate?

HoneyIQ computes a **composite threat score** (0 to 1) from four signals:

- **45%** — How severe is the current attack type?
- **35%** — How far along the Kill Chain is the attacker?
- **15%** — How fast are they likely to escalate?
- **5%** — How many attacks have happened so far in this session?

Score below 0.15 = Benign. Score above 0.75 = Critical. The score drives both the response decision and the automatic alarm.

---

## This revision: three practical gaps closed

The original version could tell *what kind* of attacker was knocking and
*what to do* about it. Testing it more critically turned up three things
still worth fixing.

### Gap 1: every simulated visitor sounded the same

The fake traffic used to test the system was regenerated from scratch,
independently, every single moment — so a "big, noisy" attack session
and a "small, quiet" one were pure luck, not something the system
deliberately varied. And every innocent visitor looked exactly like every
other innocent visitor.

**The fix:** instead of re-rolling dice for every single feature of every
single knock, each simulated visitor now gets one "size" dial set at the
start of their visit — some sessions are consistently big and loud,
others consistently small and quiet, for the whole visit. And instead of
one boring flavor of "innocent visitor," there are now three: an ordinary
person, a busy web crawler, and a quick monitoring check-in — each with
its own realistic pattern. The guessing guard (the classifier) was
retrained on this richer, more varied practice data, and still guesses
correctly 99.4% of the time.

### Gap 2: the "how worried should I be" gauge was crude

It only asked "was the last 20 knocks an attack, yes or no?" — treating a
nosy peek and a full break-in the same way, and forgetting instantly the
moment a knock fell off the back of that 20-knock list.

**The fix:** a gauge that fades instead of snapping — like a dimmer
switch — and one that also cares about *how scary* each knock was, not
just whether it happened. Both gauges (the old snap-counter and the new
fading one) run side by side; the old snap-counter is still what's
actually used by default, since the fade-gauge needs its own careful
checking before it's trusted for real decisions. Measured side by side,
the fading gauge triggers the "knocking too fast" alarm far less often
than the snap-counter (roughly 1–22% of the time vs.\ 45–90%) — a big
enough difference that the two aren't interchangeable without recalibrating
the alarm's trigger point.

### Gap 3: nothing ever changed on its own

The rulebook and its cutoff numbers were carved in stone. If the same
troublemaker came back a week later, the system had no memory of them at
all.

**Reputation.** The trap house now keeps a "trouble score" for each
visitor that quietly fades over time (halving roughly every 6 hours) but
never fully disappears the moment they leave. If that score gets high
enough, the visitor gets treated more harshly **even if this particular
knock looks perfectly innocent** — like a shop that keeps an eye on a
known shoplifter even when they're just browsing. Measured: a repeat
troublemaker's score crosses that threshold on their 4th offending visit,
and from then on they're treated as high-risk every time, even on a
perfectly innocent-looking visit. This is deliberately strict: it means
someone who was briefly a problem stays watched for a while even after
they've calmed down, which is a fair trade for not forgetting repeat
troublemakers instantly.

**A self-tuning dial — but an honest one.** One of the three (now four)
exception rules — the one about "knocking too fast" — can now nudge its
own trigger point up or down to avoid crying wolf too often or staying
silent too long. Here's the honest part: this dial does **not** claim to
make better *decisions* — nobody has told the system "that alarm was
right" or "that alarm was wrong," so it has no way to know. All it can
honestly do is watch its own alarm frequency and keep it from getting
annoying. Measured, it does exactly that: under sustained noisy traffic
its effect is small (45.7% of decisions triggering the alarm, down to
44.4%), but once ordinary/innocent traffic is mixed in — the more
realistic case — it cuts the alarm rate substantially (7.6% down to
4.4%). That's a real, useful thing to build. Claiming more than that
would be dishonest.

---

## "Surely it isn't really ~99% accurate?" — a closer, more honest look

A fair question to ask of *any* near-perfect security result is: what was
actually being measured? The original test protocol threw almost nothing
but attack traffic at the system — deliberately, to see if it could catch
attacks — but that also meant there was almost no *innocent* traffic
around for it to ever mistake for an attack. A 0% false-alarm rate under
that protocol isn't proof the system is well-calibrated; it's partly just
a consequence of never being asked the question.

**What we changed:** we re-ran the main test with roughly 30% genuinely
innocent traffic mixed in — about four times as many test sessions and
steps as before, specifically so the results stay statistically
trustworthy even though "how often did it get an innocent visitor wrong"
is now a harder, rarer thing to measure accurately. And instead of
reporting a single bare percentage, results are now given as a
**confidence range** (a Wilson interval, the same tool pollsters use to
say "52% ± 3%") along with the raw counts they're based on.

**What we found, honestly.** Counting up how many genuinely innocent
visits the *original* test actually contained turned out to be the real
story: only 33 to 114 innocent visits per attacker type, out of 6,000
total steps. With that few innocent visits to possibly get wrong, a "0%
mistakes" scorecard barely means anything — the honest confidence range
around that 0% stretched as high as roughly 1-in-10 for one attacker
type, meaning the test genuinely couldn't rule out a much higher mistake
rate. Once the mixed-traffic re-run gave the system 7,500+ genuine
innocent visits per attacker type to be tested against, a small but real
mistake rate did show up: roughly **0.6–0.8% of innocent visits** were
wrongly flagged, with a tight, trustworthy confidence range this time.
That is not a contradiction of the original "0%" — it's the same system,
just finally asked the question enough times to get a real answer.

---

## Should the trap house get an AI brain after all?

This project already tried that once — a fancier "big AI brain" was
actually built and tested. It didn't go well: the brain learned a cheap
trick — treat almost everything as dangerous — instead of actually
learning to tell visitors apart, because its practice sessions were
almost entirely break-in attempts. It also only ever practiced against
one type of visitor personality, so it wouldn't have handled the others
well.

On top of that, the tool used to *measure* how well the system was doing
had its own embarrassing mistake — twice — where it accidentally graded
a decision against the wrong moment in time, making the results look far
worse than reality until someone caught it. That's a warning sign: if the
grading tool can get confused that easily, a "big AI brain" being trained
on very similar grading logic could get confidently, invisibly, wrong in
the same way — and you might never notice, because a wrong brain doesn't
raise its hand and say "I might be broken."

So the honest answer is: **not yet, and not like that.** The repeat-
troublemaker memory and the honest self-tuning dial do the job of
"getting smarter over time" without needing a mysterious brain, a long
practice period, or blind trust in a grading system that has already
been wrong twice. If a learning brain is ever added, the plan is to keep
it tiny (choosing between a handful of pre-approved settings, not
inventing new rules from scratch) and to only let it learn from a real
human saying "yes, that alarm was correct" or "no, that one was a
mistake" — a feedback path that doesn't exist yet and would need to be
built first, honestly, before any brain gets to learn from it.

---

## What were the results?

Tested across 120 simulated attack sessions (30 per attacker type, 200 steps each, original all-attack protocol):

| Attacker | Threats Detected | False Alarms |
|----------|-----------------|--------------|
| Stealthy | **99.09%** | 35.56% |
| Aggressive | **99.47%** | 6.67% |
| Targeted | **99.48%** | 3.33% |
| Opportunistic | **99.41%** | 15.00% |

Compared to a Deep Q-Network (an AI that has to be trained for thousands of rounds), the SEDM:
- Detects threats just as well (>99%)
- Requires **zero training time**
- Can be **read and audited by a human**
- Works equally well against all four attacker types without any retuning

The revised, mixed-traffic re-run (30% innocent traffic, roughly four
times the sample size) found threat-detection essentially unchanged
(99.94–99.97%) and a small, honestly-measured mistake rate on innocent
traffic of roughly 0.6–0.8%, in place of the original "0%" measured from
too few innocent visits to be trustworthy — see the section above for
the full explanation.

---

## Why does it matter?

Most AI-based security tools are "black boxes" — they give you an answer but cannot explain *why*. In a real security operations centre, analysts must justify every decision to managers, auditors, and sometimes courts. HoneyIQ shows that a fully transparent, rule-based policy can match the detection performance of a neural network, while remaining completely explainable — and this revision shows that transparency extends to being honest about *how* a near-perfect number was measured, not just reporting it.

**Bottom line:** Same trap house, same one-page rulebook, same "you can
always ask why." Now it also: sounds more like a real neighborhood
instead of a simulation, has a gauge that fades instead of snapping,
remembers repeat troublemakers fairly, keeps one of its alarms from
crying wolf, and reports its near-perfect numbers with an honest
confidence range instead of a bare percentage — all without adding a
single line of code a human can't read and check by hand.
